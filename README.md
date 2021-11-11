<h1>Genetic Breakout</h1>
<h2>Abstract</h2>
<p>
  Breakout is a classic game created by Attari in 1976.  The goal of the game is to use a pong like ball and paddle to hit bricks at the top of the screen to break them.
  Once all breaks are broken the game advances to the next screen.
  In this version of the game, each new screen adds an additional row to increase difficulty.  Once 15 levels are completed the game ends.
  
  The Objective of this project is to use a Genetic Algorithm to train a nural network to achieve a perfect score for the game.
  Reinforcement Learning using a Genetic Algorithm to train a neural network to play a version of the classic game Breakout
  Game starts with 3 rows.
  Each time the screen is cleared an additional row is added.
  Game continues for 15 levels with a max score of 1800.
</p>


<br>The training example uses a 25 input nodes for the game state
<br>1 hidden layer with 16 nodes with ReLU activation functions
<br>and a 3 node output layer with softmax activation.

<img src="https://user-images.githubusercontent.com/94034810/141222394-a0837a16-f3ba-409c-a3da-b2d8ec996627.png" width="500">



https://user-images.githubusercontent.com/94034810/141082558-a9427023-5a3d-40bb-b583-cbd69a21dea8.mp4

<h1>Training Log</h1>
<img src="https://user-images.githubusercontent.com/94034810/141082768-7519e5b3-fba8-4f3a-a0bb-bc955b0052ff.png">
<h2>Notes:</h2>
<br>Generation 44 completes first screen
<br>Generation 72 completes perfect game (15 screens)
