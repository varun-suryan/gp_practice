clear all;
clc;
close all;
GRID = 3;
epsilon = 0.05;
gamma_ =0.9;
alpha_ = 0.8;
% Create the state action space in 1-D
states = [-GRID : 1 : GRID]';
actions = [-1; 1]; 
num_of_episodes = 50;

design_matrix = [repmat(states, numel(actions), 1) repmat(actions, numel(states), 1) 10/(1 - gamma_) * ones(numel(states) * numel(actions), 1)];


% Append initial Q-values with state action space
% design_matrix = [state_action_space zeros(size(state_action_space(:, 1)))];


% surf(stateX, stateY, stateX.^2 + stateY.^2);

l = 0.1/sqrt(2);

signal_std = 5;
noise = 0.01;

kernel_ = signal_std^2 * exp (- squareform(pdist(design_matrix(:, end - 1))).^2 / (2 * l^2) ) + noise^2 * eye(size(design_matrix,1));

mean_ = zeros(size(design_matrix, 1), 1);
% R_prior = mvnrnd(mean_, kernel_, 1);


% plot(points, R_prior,'r');

% design_matrix(:, 1 : end - 1)

% Plot the Q-Function
% surf(reshape(design_matrix(:, 1), numel(states), 2), reshape(design_matrix(:, 2), numel(states), 2), reshape(R_prior, numel(states), 2));
% hold on;


% axis([1 100 -20 20]);


% frequency = 40;
% points_train = points(1 : frequency : end);
% points_train_target = sin(points_train) + 0.05 * randn(size(points_train));


figure();
% hold on;
record = [];
plot_reward = [];

for EPISODE_number = 1 : num_of_episodes
	EPISODE = EPISODE_number
	curr_state = -1;
	total_reward = 0;
	while curr_state ~= GRID

		% Choose the action based on Optimism
		if rand <= epsilon
			action = datasample([-1 1], 1);
		else
			state_index = find(design_matrix(:, 1) == curr_state);
			temp = design_matrix(state_index, :);
			optimism = diag(kernel_);
			[M, I] = max(temp(:, 3) + 1.96 * optimism(state_index) );
			action = temp(I, 2);
		end
		
		% Calculate the next state
		next_state = max(min(curr_state + action, GRID), -GRID);

		observation = (1 - alpha_) * max(design_matrix(state_index, end)) + alpha_ * reward_dynamic(next_state, GRID) + alpha_ * gamma_ * max(design_matrix(find(design_matrix(:, 1) == next_state), end));
		total_reward = total_reward + reward_dynamic(next_state, GRID);
		record = [record; [curr_state action observation]];
		[~, indexing] = ismember(record(:, 1 : end - 1), design_matrix(:, 1 : end -1), 'rows');

		% size(record, 1)
		% size(indexing)
		% pause();
		
		coeff = pinv(kernel_(indexing, indexing) + noise^2 * eye(numel(indexing)));

		mean_ = mean_ + kernel_(:, indexing) * coeff * (record(1 : end, end) - mean_(indexing));	
		kernel_ = kernel_ - kernel_(:, indexing) * coeff * kernel_(indexing, :) + noise^2;
		% design_matrix(:, end) = mean_;
		% y_pred, sigma = gp.predict(test, return_std = True)
		curr_state = next_state;
		
		scatter(curr_state, 0);
		axis([-GRID GRID -1 1]);
		
		% hold on;
	pause(0.02)
	end
	plot_reward = [plot_reward; total_reward];
end

figure();
plot (plot_reward) 

% Sequential Implementation
% for indexing = 1 : frequency : size(points, 1);
% 	coeff = 1 / kernel_(indexing, indexing) + noise^2 * eye(numel(indexing));
% 	mean_ = mean_ + kernel_(:, indexing) * coeff * (points_train_target(count) - mean_(indexing));	
% 	kernel_ = kernel_ - kernel_(:, indexing) * coeff * kernel_(indexing, :) + noise^2;
% 	count = count + 1;
% end

% % Implementation in one Stroke
% indexing = 1 : frequency : size(points, 1);
% coeff = pinv(kernel_(indexing, indexing) + noise^2 * eye(numel(indexing)));
% mean_ = mean_ + kernel_(:, indexing) * coeff * (points_train_target - mean_(indexing));	
% kernel_ = kernel_ - kernel_(:, indexing) * coeff * kernel_(indexing, :) + noise^2;


% R_post = mvnrnd( mean_, kernel_, 1);

% plot(points, mean_, 'g')
% plot(points, R_post,'g');
% hold on;
% scatter(points_train, points_train_target)

function reward = reward_dynamic(state, GRID)
	if state == GRID
		reward = 10;
	elseif state == -GRID
		reward = -10;	
	else
		reward = -1;
	end
end
