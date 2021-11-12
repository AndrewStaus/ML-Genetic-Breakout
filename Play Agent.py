'''# Play Agent
Launch the game under control of one of the trained agents.
Graphics are enabled so you can watch it play
'''
from Train import worker
import pickle

def play_top_agent():
    '''# Play Top Agent
    Play replay for the top agent of the generation'''
    with open(r'.\trained_agents\top_agent.pickle', 'rb') as f:
        agent = pickle.load(f)
    worker(agent, graphics=True)

def play_best_agent():
    '''# Play Best Agent
    Play replay for the best agent of the training session'''
    with open(r'.\trained_agents\best_agent.pickle', 'rb') as f:
        agent = pickle.load(f)
    worker(agent, graphics=True)

def play_agent(filepath):
    '''# Play Agent
    Play replay for a saved agent
    
    ### Args:
        - filepath: filepath to an agent'''
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



