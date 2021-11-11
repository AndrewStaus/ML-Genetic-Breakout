'''Script to load a trained agent to play the game'''

from Train import worker
import pickle

def play_top_agent():
    '''play replay for the top agent of the generation'''
    with open(r'.\trained_agents\top_agent.pickle', 'rb') as f:
        agent = pickle.load(f)
    worker(agent, graphics=True)

def play_best_agent():
    '''play replay for the best agent of the training session'''
    with open(r'.\trained_agents\best_agent.pickle', 'rb') as f:
        agent = pickle.load(f)
    worker(agent, graphics=True)

def play_agent(filepath):
    '''play replay for a saved agent'''
    with open(filepath, 'rb') as f:
        agent = pickle.load(f)
    worker(agent, graphics=True)

def main():
    print('1. Pre-Trained Agent\n2. Best Agent\n3. Top Agent\n4. Exit')

    inp = input('please select: ')
    if inp == '1':
        play_agent(r'.\trained_agents\pre_trained.pickle')
    elif inp == '2':
        play_top_agent()
    elif inp == '3':
        play_best_agent()
    elif inp == '4':
        print('Goodbye!')
        return   
    else:
        print('\nSelection not supported, please choose 1,2,3,4.')
        main()

if __name__ == '__main__':

    main()



