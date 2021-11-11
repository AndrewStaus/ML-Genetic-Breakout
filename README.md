<h1>Genetic Breakout</h1>
<h2>Abstract</h2>
<p>
  Breakout is a classic game created by Attari in 1976.  The goal of the game is to use a pong like ball and paddle to hit bricks at the top of the screen to break them.
  Once all breaks are broken the game advances to the next screen.
  In this version of the game, each new screen adds an additional row to increase difficulty.  Once 15 levels are completed the game ends.
  
  The Objective of this project is to use a Genetic Algorithm to train a nural network (the agent) to achieve a perfect score for the game.
  This is a type of reinforcement learning that will "reward" the best agents by preffering them as parents when the next generation is created.
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
  
  An initial generation is created by creating a number of nural networks with random weight initilizations.
  <h3>Determine Fitness of Agents</h3>
  The Generation plays the game and their fitness is determined by the score they achieve.  If an angent gets stuck for too long without breaking a block, the game is also ended ensuring that there are no infinite loops.
  <h3>Selection</h3>
  Once fitness is determined, agents are selected stocastically weighted on their fitness.  If an agent scored higher, they are more likely to be chosen.  Agents can be chosen more than once.
  <h3>Crossover</h3>
  Once two agents are selected, crossover occurs.  Each weight and bias is given a 50% chance to be selected from either agent.  The new resulting agent has 50% of the weights from one parent, and 50% of the weights from the other.
  <h3>Mutation</h3>
  The new agent then undergoes mutation.  A small portion of the weights and biases are chosen at random for change, and then slightly tweaked.
</p>


<h2>Implementation</h2>
<h3>Libraries</h3>
<p>
  <ul>
    <li>Numpy: Vectorized calculations is key speed up hypothisis processing, greatly reducing training time</li>
    <li>Pickle: Serialize and save the agents for later use</li>
    <li>Pandas: Export log data to dataframes</li>
    <li>Pygame: Create a basic Breakout like game</li>
    <li>Concurrent Futures:  Enable multi-processing to allow mutliple agents to play at the same time durring training</li>
  </ul>
  
  <h3>Input Layer</h3>
  Input layer consists of 25 nodes:
  <ul>
    <li>Paddle X Position</li>
    <li>Ball X Position</li>
    <li>Ball Y Position</li>
    <li>Ball X Vector</li>
    <li>Ball Y Vector</li>
    <li>Count of active blocks in each row (20 values)</li>
  </ul>


  
</p>
<br>The training example uses a 25 input nodes for the game state
<br>1 hidden layer with 16 nodes with ReLU activation functions
<br>and a 3 node output layer with softmax activation.

<p align="left">
  <img src="https://user-images.githubusercontent.com/94034810/141235668-ab609c06-6714-469a-8709-47816371273e.png">
</p>



https://user-images.githubusercontent.com/94034810/141233933-9f59d17a-ec49-49fc-9114-75b2b9c29bd6.mp4




  
<h1>Training Log</h1>
<img src="https://user-images.githubusercontent.com/94034810/141082768-7519e5b3-fba8-4f3a-a0bb-bc955b0052ff.png">
<h2>Notes:</h2>
<br>Generation 44 completes first screen
<br>Generation 72 completes perfect game (15 screens)
