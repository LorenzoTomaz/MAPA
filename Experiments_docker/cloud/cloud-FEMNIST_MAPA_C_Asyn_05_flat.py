import os
from params_f import *
import _pickle as cPickle
import paho.mqtt.client as mqtt
import queue
from math import sqrt
import math
import random
import ComputePrivacy as Privacy


EPOCH = int(os.environ.get("EPOCH"))
MQTT_PORT = int(os.environ.get("MQTT_PORT"))
MQTT_IP = os.environ.get("MQTT_IP")
RESULT_ROOT = os.environ.get("RESULT_ROOT")
b = 48
msgQueue = queue.Queue()


def init_grads(params):
    grads = []
    for i in range(len(params)):
        grads.append(torch.zeros(params[i].shape))
    return grads


def Addnoise(Tensor, Clipbound):
    theta = init_grads(Tensor)
    std = z * Clipbound / K_ASyn
    for i in range(len(Tensor)):
        theta[i] = Tensor[i] + std * torch.randn(Tensor[i].shape)
    return theta


def on_connect(mqttc, obj, flags, rc):
    print("rc: " + str(rc))


def on_subscribe(client, userdata, mid, granted_qos):
    print("subscribe successful")


def on_publish(client, userdata, mid):
    print("publish success")


def on_message(mqttc, obj, msg):
    print("received: " + msg.topic + " " + str(msg.qos))
    msglist = []
    msglist.append(msg.topic)
    msglist.append(msg.payload)
    msgQueue.put(msglist)


client = mqtt.Client()
client.on_connect = on_connect
client.on_subscribe = on_subscribe
client.on_publish = on_publish
client.on_message = on_message
# client.enable_logger(logger)
client.connect(MQTT_IP, MQTT_PORT, 600)
client.subscribe("mapa_grads/#", 2)
client.loop_start()

if __name__ == "__main__":
    Layers_nodes = []
    params = InitializeParameters()
    for i in range(len(params)):
        Layers_nodes.append(params[i].numel())
    K_ASyn = 1
    # size of the dataset
    m = 93024
    # 10.1561/0400000042 page 18
    # Typically we are interested in values of δ that are less than the
    # inverse of any polynomial in the size of the database.
    # In particular, values of δ on the order of 1/∥x∥ are very dangerous:
    # they permit “pre- serving privacy” by publishing the complete records
    # of a small number of database participants — precisely the “just a few”
    # philosophy discussed in Section 1.
    delta = 1 / (m**1.1)
    # variance used for gaussian noise
    z = 0.5
    # number of iterations
    T = 1800 * 3
    # upper bound of ‖∇F(x, ξ)‖ (∇F(x, ξ): gradient computed on a sample ξ) - noise level
    G = torch.tensor([0.23])
    # variance of ∇F(x, ξ) (∇F(x, ξ): gradient computed on a sample ξ)
    SIGMA = 5
    # Lipschitz smooth constant for which ‖∇xF(x, ξ)−∇xF(y, ξ)‖∗≤L‖x−y‖
    L = 0.021
    # θ ∈(0,1) is used to control the reduction ratio.
    theta = torch.tensor([0.8])
    # space radius?? maximum norm of the parameters across all devices?
    R = 0.1
    # maximal delay
    tau = torch.tensor([3.0])
    # variance of the noise added for privacy
    Delta_b = (b * SIGMA**2 + sum(Layers_nodes) * (z * G) ** 2) / b
    # an undetermined coefficient and P≥1
    P = max(2 * Delta_b / (theta**2 * G**2 * (tau + 1)), 1)
    # learning rate
    LR = 1 / (2 * P * L * (tau + 1))
    # number of global updates that have to be completed before the noise parameter G is reduced
    T0 = max(math.ceil(4 * P**2 * L * (tau + 1) ** 2 * R / Delta_b), 1)
    payload = [z, G, params]
    client.publish("init", cPickle.dumps(payload), 2)

    for epoch in range(EPOCH):
        for step in range(T + 1):
            print("step:", step, "LR:", LR, "G:", G, "T0:", T0)
            edge_topic = []
            grads_sum = init_grads(params)
            for i in range(K_ASyn):
                msglist = msgQueue.get()
                edgetopic = msglist[0]
                edgemsg = msglist[1]
                edgetopic = "mapa_params/" + edgetopic.split("/")[1]
                edge_topic.append(edgetopic)
                message_ = cPickle.loads(edgemsg)
                grads = message_[0]
                Clipbound = message_[1]
                for i in range(len(grads)):
                    grads_sum[i] += grads[i]
            for i in range(len(grads_sum)):
                grads_sum[i] /= K_ASyn

            grads_noise = Addnoise(grads_sum, Clipbound)
            for i in range(len(params)):
                params[i] = params[i].float()
                params[i] -= LR * grads_noise[i]

            payload = [z, G, params]
            for i in range(len(edge_topic)):
                client.publish(edge_topic[i], cPickle.dumps(payload), 2)

            if step % 30 == 0:
                man_file = open(RESULT_ROOT + "[MAPA_Budget]" + ".txt", "a")
                varepsilon = Privacy.ComputePrivacy(b / m, z, step + 1, delta, 32)
                man_file.write(str(varepsilon) + "\n")
                man_file.close()

            if epoch * (m / BATCH_SIZE) + step >= T0:
                G = G * theta
                Delta_b = (SIGMA**2 + sum(Layers_nodes) * (z * G) ** 2) / b
                P = max(2 * Delta_b / (theta**2 * G**2 * (tau + 1)), 1)
                LR = 1 / (2 * P * L * (tau + 1))
                T0 = (
                    max(math.ceil(4 * P**2 * L * (tau + 1) ** 2 * R / Delta_b), 1)
                    + T0
                )
                continue

            if epoch * (m / BATCH_SIZE) + step == T:
                break
