from training import worker
import pickle

if __name__ == '__main__':

    with open(r'.\Trained Agents\1410 trained_agent_16_Relu.pickle', 'rb') as f:
        agent = pickle.load(f)

    worker(agent, graphics=True)
