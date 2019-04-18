% chen chen 03/18/2019
% backpropagation(3 layeres) without neural networ toolbox
% accuracy 89%
% perhaps need to initialise a few times to see whether the starting point
% is within the range.

clear, close all


%%
test=hdf5info('test_128.h5');
test= hdf5read(test.GroupHierarchy.Datasets)';
x=hdf5info('train_128.h5');
x= hdf5read(x.GroupHierarchy.Datasets)';
t=hdf5info('train_label.h5');
t= double(hdf5read(t.GroupHierarchy.Datasets));
t=(t>=5);
xtest=x(50001:60000,:);
ttest=t(50001:60000,:);
x=x(1:50000,:);
t=t(1:50000,:);

% acquire number of training patterns and feature vectors
[numTP, numFV] = size(x);

% output layer's unit size
numOut = size(t,2);

%x = (x-mean(x));
%x = x./std(x);

% add 1 as bias
x = [ones(numTP,1), x];
xtest = [ones(10000,1), xtest];
x_min = min(min(x));
x_max = max(max(x));

% learning
lr = 0.1;   % learning rate
max_iteration = 80;    
numHid = 100; % hidden(midle) layer's unit size

% init
mse = 1;
w1_new = zeros(numHid, numFV + 1, numTP);
w2_new = zeros(numOut, numHid + 1, numTP);

% weight value range[-1-1]
w1 = 2 * rand(numHid, numFV + 1) - 1;
w2 = 2 * rand(numOut, numHid + 1) - 1;

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
    mse(1,iteration) = sum(sum((o-t).^2)')/(numTP*numOut)
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
% calculate hidden layer
z = 1 ./ (1 + exp(-w1 * xtest'))';
% cauculate output layer
o = 1 ./ (1 + exp(-w2 * [ones(1,10000); z']))';
accuracy = sum(round(o)==ttest) / 10000 * 100