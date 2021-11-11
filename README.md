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
  
  <h3>Determine Fitness of Agents</h3>
  <h3>Selection</h3>
  <h3>Crossover</h3>
  <h3>Mutation</h3>
</p>



<br>The training example uses a 25 input nodes for the game state
<br>1 hidden layer with 16 nodes with ReLU activation functions
<br>and a 3 node output layer with softmax activation.

<p align="center">
  <img src="(https://user-images.githubusercontent.com/94034810/141234716-ea5c7765-de94-4740-ab7c-1125e6b3d3eb.png" width="500">
</p>







https://user-images.githubusercontent.com/94034810/141233933-9f59d17a-ec49-49fc-9114-75b2b9c29bd6.mp4




  
<h1>Training Log</h1>
<img src="https://user-images.githubusercontent.com/94034810/141082768-7519e5b3-fba8-4f3a-a0bb-bc955b0052ff.png">
<h2>Notes:</h2>
<br>Generation 44 completes first screen
<br>Generation 72 completes perfect game (15 screens)
