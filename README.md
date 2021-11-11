<h1>Genetic Breakout</h1>
<h2>Abstract</h2>
<p>
	Breakout is a classic game created by Atari in 1976.  The goal of the game is to use a <i>Pong</i> like ball and paddle to hit bricks
	at the top of the screen to break them.  Breaking a break awards a point.
	Once all bricks are broken the game advances to the next screen.
	In this version of the game, each new screen adds an additional row of bricks to increase difficulty.  Once 15 levels are completed the game ends.
</p>
<p>
	The objective of this project is to use a Genetic Algorithm to train a neural network (the agent) to achieve a perfect score for the game (1800 points).
	I will use a type of unsupervised learning called reinforcement learning that will "reward" the best agents by preferring them as parents
	when the next generation is created.
</p>

<h2>The Genetic Algorithm</h2>
<p>
	A genetic algorithm will be used in this implementation of reinforcement learning.
	Genetic algorithms typically have four main subroutines:
	<ol>
		<li>Determine Fitness of Agents</li>
		<li>Selection</li>
		<li>Crossover</li>
		<li>Mutation</li>
	</ol>
	These steps ensures that the agent pool stays diverse so that new approaches are tried, and the agents do not become stuck in local optima.
</p>

<h3>Determine Fitness of Agents</h3>
<p>
	Fitness is a measure of how well an agent performed a desired task.  If a fitness function does not accurately reward desired activity,
	agents may be slow to learn or not even 
	progress.
	To determine fitness, all agents in a generation play the game and their fitness is determined by the score they achieve.
	If an agent gets stuck for too long without breaking 
	a block, the agent loses a life ensuring that there is not an incentive for the agents to enter infinite loops.
</p>
<h3>Selection</h3>
<p>
	Once fitness is determined, agents are selected stochastically, weighted by their fitness.  Agents with higher
	fitness are more likely to be chosen more often.  Agents can be chosen more than once.
</p>
<h3>Crossover</h3>
<p>
	Once two agents <i>(parents)</i> are selected, crossover occurs.  Each weight and bias are given a 50% chance to
	be selected from either parent.  The new resulting agent <i>(child)</i> has about half of the weights from one parent,
	and about half of the weights from the other.
</p>
<h3>Mutation</h3>
<p>
	The new agent <i>(child)</i> then undergoes mutation.  A small portion of the weights and biases are chosen at random for change, and then slightly tweaked.
</p>


<h2>Implementation</h2>
<h3>Libraries</h3>
<p>
	<ul>
		<li><b>Numpy</b>: Vectorized calculations speed up hypothesis processing, greatly reducing training time</li>
		<li><b>Pickle:</b> Serialize and save the agents for later use</li>
		<li><b>Pandas:</b> Export log data to data frames</li>
		<li><b>Pygame:</b> Create a basic Breakout like game</li>
		<li><b>Concurrent Futures:</b>  Enable multi-processing to allow multiple agents to play at the same time during training</li>
	</ul>
</p>

<h3>Layers</h3>
<p>
	<b>Input layer</b> consists of 25 nodes:
	<ul>
		<li>Paddle X Position</li>
		<li>Ball X Position</li>
		<li>Ball Y Position</li>
		<li>Ball X Vector</li>
		<li>Ball Y Vector</li>
		<li>Count of active blocks in each row (20 values)</li>
	</ul>
</p>
<p>
	<b>Hidden layer</b> has 16 nodes using ReLU activations.
	<br><b>Output layer</b> has 3 nodes with Softmax activations:
	<ul>
		<li>Move Paddle Left</li>
		<li>Move Paddle Right</li>
		<li>Do Nothing</li>
	</ul>
</p>
<p>
	For each game clock cycle <i>(frame)</i> the agent is given the input and makes a decision what action to take <i>(hypothesis)</i>.
	Softmax activation ensures that conflicting actions are not possible.
</p>
<p align="center">
	<img src="https://user-images.githubusercontent.com/94034810/141366921-9dd698c9-7ddc-473f-bc22-68b6f2b36cc6.png" title="Network Design">
</p>

<h2>Training</h2>
<p>
	256 agents are tested for fitness in each generation.  To speed up training time, multiple agents play the game in in parallel by leveraging multi-processing:
</p>
<p align="center">
	<img src="https://user-images.githubusercontent.com/94034810/141365553-86b42305-2977-417d-b4b0-b9eb9f220b9d.png", title="Multi-Processing">
</p>
<p>
	Because there are so many games running simultaneously system resources are taxed by rending the graphics.
	To further speed up training time, the games are run in headless 
	mode with graphics and interface disabled so that the CPU resources can be used more efficiently.
</p>


<h3>Hyper-Parameters</h3>
<p>
	<ul>
		<li><b>Fitness:</b> Score^2.  Using an exponential function for fitness causes agents that perform slightly better
			to have a much larger probability in selected than their close competitors.  This helps keep the agent pool healthy</li>
		<li><b>Mutation Rate:</b> 25% of the weights and biases will be altered on an agent during the mutation step</li>
		<li><b>Mutation Scale:</b> 0.10 will cause a relatively small change to occur  on the weights and biases that are chosen randomly for mutation</li>
	</ul>
</p>

<h3>Results</h3>
<p>
	Optimization was slow for the first 44 generations while the agents learned how to clear the first screen.  Once that hurdle was overcome,
	they were able to generalize to later levels spiking the learning rate.
</p>
<p align="center">
	<img src="https://user-images.githubusercontent.com/94034810/141082768-7519e5b3-fba8-4f3a-a0bb-bc955b0052ff.png" title="Results Log">
</p>

<p>
	Top agent scores fluctuate during training, but the mean score of the population continues to increase.  There is a breakthrough around generation 60 and the agents
	optimize for a perfect score on generation 72.
</p>

<h2>Trained Agent Playback</h2>
<p>
	The below video shows the playback of the trained agent successfully completing level 15 and achieve the max score of 1800 without losing any lives.
</p>

https://user-images.githubusercontent.com/94034810/141233933-9f59d17a-ec49-49fc-9114-75b2b9c29bd6.mp4

<p>
Playback is not sped up; the framerate is bottlenecked mainly by the hypothesis calculation time of the agent.
</p>

<h2>Conclusions</h2>
<p>
	The goal of the project was achieved!  An agent that is capable of scoring a perfect game was trained in 72 generations,
	however the agent does not play optimally. There are long periods of not scoring any points before finding the brick when
	the screen is almost cleared.  There is room for improvement.
</p>
<h3>Possible Enhancements</h3>
<p>
	<ul>
		<li><b>Alter fitness function:</b>
			The current fitness function is only based off high scores.  Indirectly the time limit imposed on the agent to make a score does influence 
			them to not waste time, but a bonus score for screen clear times would likely improve performance</li>
		<li><b>Add more inputs:</b> The agent's knowledge of the game state is limited.
			It only knows the number of active blocks in a row, it does not know what column the blocks 
			are in.  Increasing the number of inputs would likely help the network but would also greatly
			increase training time as there will be many more weights and biases.</li>
		<li><b>Add more hidden layers or nodes:</b> a single 16 node hidden layer is quite shallow.
			Increasing the complexity may allow the agents to learn more sophisticated 
			functions.  However, this would also greatly increase training times.</li>
	</ul>
</p>

<h1>File Descriptions</h1>
<p>
	<ul>
		<li><b>Breakout.py: </b>This is the human playable game.  The agents make it look easy, give it a go for yourself!</li>
		<li><b>Train.py: </b>This is the main training script.  It will train a new batch of agents.  Results will be printed to the console and logged</li>
		<li><b>Play Agent.py:</b> This will launch the game under control of one of the trained agents.  Graphics are enabled so you can watch it play.</li>
		<li><b>Result Notebook.py:</b> Outputs the log to a graph so you can have a visual reference of training performance.
			Can be accessed while training is in progress</li>
	</ul>
</p>
