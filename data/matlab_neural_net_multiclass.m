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
y=hdf5info('train_label.h5');
y= double(hdf5read(y.GroupHierarchy.Datasets));

xtest=x(50001:60000,:);
ttest=y(50001:60000,:);
x=x(1:50000,:);
y=y(1:50000,:);


% acquire number of training patterns and feature vectors
[numTP, numFV] = size(x);

% output layer's unit size
numOut = max(y)+1;
%one hot encoding
y = y == 0:max(y);

x = (x-mean(x));
x = x./std(x);

% add 1 as bias
x = [ones(numTP,1), x];
xtest = [ones(10000,1), xtest];
x_min = min(min(x));
x_max = max(max(x));

% learning
lr = .1;   % learning rate
max_iteration = 80;    
numHid = 100; % hidden(midle) layer's unit size

% init
mse = 1;
w1_new = zeros(numHid, numFV + 1, numTP);
w2_new = zeros(numOut, numHid + 1, numTP);

% weight value range[-1-1]
w1 = 2 * rand(numHid, numFV + 1) - 1;
w2 = 2 * rand(numOut, numHid + 1) - 1;
%w1 = w1/100
%w2 = w2/100

% 
for iteration = 1 : max_iteration
    
    %
    % calculate hidden layer
    z2 = 1 ./ (1 + exp(-w1 * x'))';
    % cauculate output layer
    z1 = 1 ./ (1 + exp(-w2 * [ones(1,numTP); z2']))';
    %for i=1:9
    %    o = 1 ./ (1 + exp(-w2 * [ones(1,numTP); z']))';
    %end
    a = softmax(z1')';
    
    dLdo = (-y./a);
    aa=permute(a,[2 3 1]);
    yy=permute(y,[3 2 1]);
    aa=repmat(aa,[1 numOut 1]);
    yy=repmat(yy,[numOut 1 1]);
    ay=aa.*yy;
    ay2=permute(sum(ay,2),[3 1 2])+y;
    
    % calculate gragient output layer
    delta2 = ay2;
    %delta2 = (y - a) .* a .* (1 - a);
    % calculate gragient hidden layer
    delta1 = z2 .* (1 - z2) .* (delta2 * w2(:,2:end));
    
    % sum of training pattern
    w2_new = lr * ay2' * [ones(numTP,1), z2];
    w1_new = lr * delta1' * x;    
    
    % update w2
    w2 = w2 + w2_new;
    % update w2
    w1 = w1 + w1_new;
        
    % mean square error
    %mse(1,iteration) = sum(sum((o-t).^2)')/(numTP*numOut)
    [~,i]=max(a,[],2);
    sum(i==(y*(1:10)'))/50000
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
z2 = 1 ./ (1 + exp(-w1 * xtest'))';
% cauculate output layer
a = 1 ./ (1 + exp(-w2 * [ones(1,10000); z2']))';
[~,i]=max(a,[],2);

accuracy = sum(round(i)==ttest) / 10000 * 100