clear all;
clc;
close all;
GRID = 1;
% Create the state action space in 2-D
[stateX, stateY] = meshgrid(-GRID: 1 :GRID, -GRID: 1 :GRID);
actions = [0 1; 1 0; 0 -1; -1 0]; 
state_action_space = [repmat([stateX(:) stateY(:)], size(actions, 1), 1) repmat(actions, numel(stateX), 1)];


% Append initial Q-values with state action space
design_matrix = [state_action_space zeros(size(state_action_space(:, 1)))];


% surf(stateX, stateY, stateX.^2 + stateY.^2);

l = 6/sqrt(2);

signal_std = 2;
noise = 0.1;

kernel_ = signal_std^2 * exp (- squareform(pdist(state_action_space)).^2 / (2 * l^2) ) + noise^2 * eye(size(state_action_space,1));

mean_ = zeros(size(state_action_space,1), 1);
R_prior = mvnrnd(mean_ , kernel_, 1);

pause();

plot(points, R_prior,'r');
hold on;
% axis([1 100 -20 20]);


frequency = 40;
points_train = points(1 : frequency : end);
points_train_target = sin(points_train) + 0.05 * randn(size(points_train));





% for time_step = 1 : 100
	
% 	action = chooseAction(curr_state)

	
% 	next_state = (max(min(curr_state(1) + action(1), GRID), -GRID) , max(min(curr_state(2) + action(2), GRID), -GRID))
	
% 	design_mat_t.append(reward_dynamics(next_state) + gamma * max([getQ(next_state, a) for a in actions]))
	

% 	y_pred, sigma = gp.predict(test, return_std = True)

% 	curr_state = next_state
% end


% Sequential Implementation
count = 1;
for indexing = 1 : frequency : size(points, 1);
	coeff = 1 / kernel_(indexing, indexing) + noise^2 * eye(numel(indexing));
	mean_ = mean_ + kernel_(:, indexing) * coeff * (points_train_target(count) - mean_(indexing));	
	kernel_ = kernel_ - kernel_(:, indexing) * coeff * kernel_(indexing, :) + noise^2;
	count = count + 1;
end

% % Implementation in one Stroke
% indexing = 1 : frequency : size(points, 1);
% coeff = pinv(kernel_(indexing, indexing) + noise^2 * eye(numel(indexing)));
% mean_ = mean_ + kernel_(:, indexing) * coeff * (points_train_target - mean_(indexing));	
% kernel_ = kernel_ - kernel_(:, indexing) * coeff * kernel_(indexing, :) + noise^2;


% R_post = mvnrnd( mean_, kernel_, 1);

plot(points, mean_, 'g')
% plot(points, R_post,'g');
hold on;
scatter(points_train, points_train_target)

