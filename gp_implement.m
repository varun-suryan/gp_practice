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

design_matrix = [repmat(states, numel(actions), 1) repmat(actions, numel(states), 1) zeros(numel(states) * numel(actions), 1)];


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




for i = 1 : 20

	curr_state = 0;
	while curr_state ~= GRID
		if rand <= epsilon
			action = datasample([-1 1], 1);
		else
			state_index = find(design_matrix(:, 1) == curr_state);
			temp = design_matrix(state_index, :);

			optimism = diag(kernel_)
			[M, I] = max(temp(:, 3) + 1.96 * optimism(state_index) );
			action = temp(I, 2);
		end

	% next_state = (max(min(curr_state(1) + action(1), GRID), -GRID) , max(min(curr_state(2) + action(2), GRID), -GRID));
		next_state = max(min(curr_state + action, GRID), -GRID);
		[~, indexing, ~] = intersect(design_matrix(:, 1 : end -1), [curr_state action], 'rows');

		observation = (1 - alpha_) * design_matrix(indexing, end) + alpha_ * reward_dynamic(next_state, GRID) + alpha_ * gamma_ * max(design_matrix(find(design_matrix(:, 1) == next_state), end));
	
		coeff = 1 / (kernel_(indexing, indexing) + noise^2 * eye(numel(indexing)));

		mean_ = mean_ + kernel_(:, indexing) * coeff * (observation - mean_(indexing));	
		kernel_ = kernel_ - kernel_(:, indexing) * coeff * kernel_(indexing, :) + noise^2;
		design_matrix(:, end) = mean_;
		% y_pred, sigma = gp.predict(test, return_std = True)
		design_matrix	
		curr_state = next_state
	pause()
	end
end

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
		reward = -10	
	else
		reward = -1;
	end
end
