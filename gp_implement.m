clear all;
clc;
close all;
GRID = 25;
epsilon = 0.10;
gamma_ =0.9;
alpha_ = 0.6;


% v = VideoWriter('q_learning');
% open(v);

% Create the state action space in 2-D
[stateX, stateY] = meshgrid(-GRID: 1 :GRID, -GRID: 1 :GRID);
actions = [0 1; 1 0; 0 -1; -1 0]; 
state_action_space = [repmat([stateX(:) stateY(:)], size(actions, 1), 1) repmat(actions, numel(stateX), 1)];

design_matrix = [state_action_space 100 * ones(numel(stateX) * size(actions, 1), 1)];

% Append initial Q-values with state action space
% design_matrix = [state_action_space zeros(size(state_action_space(:, 1)))];


% surf(stateX, stateY, stateX.^2 + stateY.^2);

% l = 1/sqrt(2);

% signal_std = 5;
% noise = 0.1;

% kernel_ = signal_std^2 * exp (- squareform(pdist(design_matrix(:, end - 1))).^2 / (2 * l^2) ) + noise^2 * eye(size(design_matrix,1));

% mean_ = zeros(size(design_matrix, 1), 1);

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

for episode = 1 : 200
Episode = episode
% curr_state = [round(2 * GRID * (rand - 0.5)) 2 * round(GRID * (rand - 0.5))]; 
curr_state = [-GRID -GRID];
% count = 0;
	while ~isequal(curr_state, [GRID GRID]) 
		% count = count + 1;
		% if count > 200
		% 	break;
		% end

		if rand <= epsilon
			action = actions(randsample(4, 1), :);
		else
			temp = design_matrix(find(ismember(design_matrix(:, 1:2), curr_state, 'rows')), :);
			I = datasample(find(temp(:, end) == max(temp(:, end))), 1);
			action = temp(I, 3 : 4);
		end	

		next_state = [max(min(curr_state(1) + action(1), GRID), -GRID)  max(min(curr_state(2) + action(2), GRID), -GRID)];

		[~, indexing, ~] = intersect(design_matrix(:, 1 : end -1), [curr_state action], 'rows');	

		argmax = max(design_matrix(find(ismember(design_matrix(:, 1:2), next_state, 'rows')), :));
		observation = (1 - alpha_) * design_matrix(indexing, end) + alpha_ * reward_dynamic(next_state, action, GRID) + alpha_ * gamma_ * argmax(end);
		design_matrix(indexing, end) = observation;	

		% scatter(curr_state(1), curr_state(2), 100, 'filled');
		% axis([-GRID GRID -GRID GRID]);
		% frame = getframe(gcf);
  %  		writeVideo(v,frame);
		% grid on;

		curr_state = next_state;
		
		% pause(0.01);
	end
end
close(v);

function reward = reward_dynamic(state, action, GRID)
	if isequal(state, [GRID GRID])
		reward = 100;
	else
		reward = 10 * sum(action) + randn;
	end
end
