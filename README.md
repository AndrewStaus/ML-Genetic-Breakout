<h1>Genetic Breakout</h1>
<h2>Goal</h2>
<p>
  Breakout is a classic game created by <i>Attari</i> in 1976.  The goal of the game is to use a <i>Pong</i> like ball and paddle to hit bricks at the top of the screen to break them.  Breaking a break awards a point.
  Once all breaks are broken the game advances to the next screen.
  In this version of the game, each new screen adds an additional row of bricks to increase difficulty.  Once 15 levels are completed the game ends.
  
  The Objective of this project is to use a Genetic Algorithm to train a neural network (the agent) to achieve a perfect score for the game.
  This is a type of unsupervised learning called <i>reinforcement learning</i> that will "reward" the best agents by preferring them as parents when the next generation is created.
</p>

<h2>The Genetic Algorithm</h2>
<p>
  The algorithm being used is broken into 4 steps
  <ul>
    <li>Determine Fitness of Agents</li>
    <li>Selection</li>
    <li>Crossover</li>
    <li>Mutation</li>
  </ul>
  These steps ensures that the agent pool stays diverse so that new approaches are tried, and the agents do not become stuck in local optima.
  

  <h3>Determine Fitness of Agents</h3>
  Fitness is a measure of how well an agent performed a desired task.  If a fitness function does not accurately reward desired activity, agents may be slow to learn or not ever progress.
  To determine fitness the all agents in a generation play the game and their fitness is determined by the score they achieve.  If an agent gets stuck for too long without breaking a block, the agent loses a life ensuring that there is not an incentive for the agents to enter infinite loops.
  <h3>Selection</h3>
  Once fitness is determined, agents are selected stochastically, weighted on their fitness.  Agents with higher fitness are more likely to be chosen more often.  Agents can be chosen more than once.
  <h3>Crossover</h3>
  Once two agents <i>(parents)</i> are selected, crossover occurs.  Each weight and bias are given a 50% chance to be selected from either agent.  The new resulting agent <i>(child)</i> has 50% of the weights from one parent, and 50% of the weights from the other.
  <h3>Mutation</h3>
  The new agent <i>(child)</i> then undergoes mutation.  A small portion of the weights and biases are chosen at random for change, and then slightly tweaked.
</p>


<h2>Implementation</h2>
<h3>Libraries</h3>
<p>
  <ul>
    <li><b>Numpy</b>: Vectorized calculations is key speed up hypothesis  processing, greatly reducing training time</li>
    <li><b>Pickle:</b> Serialize and save the agents for later use</li>
    <li><b>Pandas:</b> Export log data to data frames</li>
    <li><b>Pygame:</b> Create a basic Breakout like game</li>
    <li><b>Concurrent Futures:</b>  Enable multi-processing to allow multiple  agents to play at the same time during training</li>
  </ul>
  
  <h3>Layers</h3>
  <b>Input layer</b> consists of 25 nodes:
  <ul>
    <li>Paddle X Position</li>
    <li>Ball X Position</li>
    <li>Ball Y Position</li>
    <li>Ball X Vector</li>
    <li>Ball Y Vector</li>
    <li>Count of active blocks in each row (20 values)</li>
  </ul>
  
  <b>Hidden layer</b> has 16 nodes using ReLU activations
  
  <b>Output layer</b> has 3 nodes with Softmax activations:
  <ul>
  <li>Move Paddle Left ('<-' input)</li>
  <li>Move Paddle Right ('->' input)</li>
  <li>Do Nothing</li>
  </ul>
  
  For each frame the network is given the input and makes a decision what action to take.  Softmax activation ensures that conflicting actions are not taken.
  <img src="https://user-images.githubusercontent.com/94034810/141235668-ab609c06-6714-469a-8709-47816371273e.png">
</p>

<h2>Training</h2>
<p>
  256 agents are tested for fitness in each generation.  To speed up testing, games are run in parallel with graphics disabled.  Disabling graphics greatly increases the processing of each frame, greatly reducing run time.
  
  <h3>Hyper-Parameters</h3>
  <ul>
  <li><b>Fitness:</b> Score^2.  Using an exponential function for fitness causes agents that perform slightly better to have a much larger probability in selected than their close competitors.  This helps keep the agent pool healthy</li>
  <li><b>Muration Rate:</b> 25% of the weights and biases will be altered on an agent during the mutation step</li>
  <li><b>Mutation Scale:</b> 0.10, this will cause a relatively small change to occur  on the weights and biases that are chosen randomly for mutation</li>
  </ul>
  
  
  <h3>Results</h3>
  Optimization was slow for the first 44 generations while the agents learned how to clear the first screen.  Once that hurdle was overcome, they were able to generalize to later levels spiking the learning rate.
  <img src="https://user-images.githubusercontent.com/94034810/141082768-7519e5b3-fba8-4f3a-a0bb-bc955b0052ff.png">
  Top agent scores fluctuate during training, but the mean score of the population continues to increase.  There is a breakthrough around generation 60 and the agents are able to optimize for a perfect score on generation 72.
</p>

<h2>Trained Agent Playing</h2>

https://user-images.githubusercontent.com/94034810/141233933-9f59d17a-ec49-49fc-9114-75b2b9c29bd6.mp4

<h2>Postmortem</h2>
<p>
  While training was realativly fast --only taking 72 generations-- the agent does not play optimally.
  Some possible solutions:
  <ul>
    <li><b>Alter Fitness Function:</b> The current fitness function is only based off high scores.  Indirectly the time limit imposed on the agent to make a score does influence them to not waste time, but a bonus score for screen clear times would likely improve performance</li>
    <li><b>Add more inputs:</b> The agents knowledge of the gamestate is limited.  It only knows the number of active blocks in a row, it does not know what column the blocks are in.  Increasing the number of inputs would likely help the network, but would also greatly increase training time as there will be many more weights and biases.</li>
    <li><b>Add more hidden layers or nodes:</b> a single 16 node hidden layer is quite shallow.  Increasing the complexity may allow the agents to learn more sophisticated functions.  However, this would also greatly increase training times.</li>
  </ul>
</p>

<h1>File Descriptions</h1>
<p>
  <ul>
    <li><b>Breakout.py: </b>This is the human playable game.  The agents make it look easy, give it a go for yourself!</li>
    <li><b>Train.py: </b>This is the main training script.  It will train a new batch of agents.  Results will be printed to the console and logged</li>
    <li><b>Play Agent.py:</b> This will launch the game under control of one of the trained agents.  Graphics are enabled so you can watch it play.</li>
    <li><b>Result Notebook.py:</b> Outputs the log to a graph so you can have a visual reference of training performance.  Can be accessed while training is in progress</li>
  </ul>
</p>
