
clear, close all
N = 50;
noise = 0.45;
% exclusive or(XOR) probrem
x1 = [noise * rand(N,1) - noise/2, noise * rand(N,1) + noise*2];
x2 = [noise * rand(N,1) + noise*2, noise * rand(N,1) - noise/2];
x3 = [noise * rand(N,1) - noise/2, noise * rand(N,1) - noise/2];
x4 = [noise * rand(N,1) + noise*2, noise * rand(N,1) + noise*2];
figure(1)
plot(x1(:,1),x1(:,2),'rs', x2(:,1),x2(:,2),'rs', x3(:,1),x3(:,2),'bs', x4(:,1),x4(:,2),'bs');
axis square
axis([-0.5 1.5 -0.5 1.5])
title('Training data')
xlabel('Feature 1')
ylabel('Feature 2')
%%
x = [x1; x2; x3; x4];   % feature vector 
t = [ones(N * 2, 1); zeros(N * 2 ,1)];  % desired value
% acquire number of training patterns and feature vectors
[numTP, numFV] = size(x);
% output layer's unit size
numOut = size(t,2);
% add 1 as bias
x = [ones(numTP,1), x];
x_min = min(min(x));
x_max = max(max(x));
% learning
lr = 0.1;   % learning rate
max_iteration = 100;    
numHid = 5; % hidden(midle) layer's unit size
% init
mse = 1;
w1_new = zeros(numHid, numFV + 1, numTP);
w2_new = zeros(numOut, numHid + 1, numTP);
% weight value range[-1-1]
w1 = 2 * rand(numHid, numFV + 1) - 1; % 5 3
w2 = 2 * rand(numOut, numHid + 1) - 1; % 1 6
% 
for iteration = 1 : max_iteration
    
    %
    
    % calculate hidden layer
    z = 1 ./ (1 + exp(-w1 * x'))';
    % cauculate output layer
    o = 1 ./ (1 + exp(-w2 * [ones(1,numTP); z']))';

    % calculate gragient output layer
    delta2 = (t - o) .* o .* (1 - o);
    % calculate gragient hidden layer
    delta1 = z .* (1 - z) .* (delta2 * w2(:,2:end));
    
    % sum of training pattern
    w2_new = lr * delta2' * [ones(numTP,1), z];
    w1_new = lr * delta1' * x;    
    
    % update w2
    w2 = w2 + w2_new;
    % update w2
    w1 = w1 + w1_new;
        
    % mean square error
    mse(1,iteration) = sum(sum((o-t).^2)')/(numTP*numOut);
    iteration
    
end
% visualize learning
figure(2)
subplot(4,1,1)
imagesc(w1(:,2:end)', [min(min(w1(:,2:end))), max(max(w1(:,2:end)))]);
title('input-hidden weight')
ylabel('input layer')
xlabel('hidden layer')
ax = gca;
ax.XTick = 1:numHid;
ax.YTick = 1:numFV;
colorbar
subplot(4,1,2)
imagesc(w2(:,2:end), [min(min(w2(:,2:end))), max(max(w2(:,2:end)))]);
title('hidden-output weight')
xlabel('hidden layer')
ylabel('output layer')
ax = gca;
ax.XTick = 1:1:numHid;
ax.YTick = 1:1:numOut;
colormap(hot);
colorbar
%
subplot(2,1,2)
plot(1:max_iteration, mse, 'k');
hold on;
grid
title('Learning curve')
xlabel('iteration');
ylabel('mse')
%% plot map and decision boundary
d = 4e-3;
[grid_x1, grid_x2] = meshgrid(round(min(x(:,2)))-1:d:round(max(x(:,3)))+1, round(min(x(:,3)))-1:d:round(max(x(:,2)))+1);
maxPixel = length(grid_x1(1,:)) * length(grid_x2(:,1));
for j = 1:length(grid_x2(:,1))
    featureSpace((j-1)*length(grid_x1(1,:))+1:length(grid_x1(1,:))*j,:) = [grid_x1(1,:)', ones(length(grid_x1(1,:)),1) * grid_x2(j,1)];
end
% generate unseen data and calculate
unseen_x = featureSpace;
unseen_x = [ones(maxPixel,1), unseen_x];
unseen_v = round(1 ./ (1 + exp(-w1 * unseen_x')));
unseen_o = 1 ./ (1 + exp( -w2 * [ones(1,maxPixel); unseen_v]));
value_grid = reshape(unseen_o, length(grid_x1(1,:)), length(grid_x1(:,1)));
figure(3)
colormap(jet)
imagesc(grid_x2(:,1), grid_x1(1,:), round(value_grid),[0,1])
hold on
contour(grid_x2(:,1), grid_x1(1,:), round(value_grid),1)
set(gca,'YDir','normal')
hold on
plot(x1(:,1),x1(:,2),'rs',x2(:,1),x2(:,2),'rs',x3(:,1),x3(:,2),'bs',x4(:,1),x4(:,2),'bs');
axis square
axis([-0.5 1.5 -0.5 1.5])
title('Decision boundary')
xlabel('Feature 1')
ylabel('Feature 2')
accuracy = sum(round(o)==t) / numTP * 100