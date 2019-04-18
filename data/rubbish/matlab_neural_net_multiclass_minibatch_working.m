% chen chen 03/18/2019
% backpropagation(3 layeres) without neural networ toolbox
% accuracy 83.6%
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

trainsize=50048;
xtest=x((trainsize+1):60000,:);
ytest=y((trainsize+1):60000,:);
x=x(1:trainsize,:);
y=y(1:trainsize,:);


% acquire number of training patterns and feature vectors
[numTP, numFV] = size(x);

% output layer's unit size
numOut = max(y)+1;
%one hot encoding
y = y == 0:max(y);

%x = (x-mean(x));
%x = x./std(x);


% add 1 as bias


x_min = min(min(x));
x_max = max(max(x));

% learning
lr = 0.1;   % learning rate
max_iteration = 5;    
numHid = 32; % hidden(midle) layer's unit size

% init
loss = 1 : max_iteration;
w1_new = zeros(numHid, numFV + 1, numTP);
w2_new = zeros(numOut, numHid + 1, numTP);

% weight value range[-1-1]
w1 = 2 * rand(numHid, numFV + 1) - 1;
w2 = 2 * rand(numOut, numHid + 1) - 1;
%w1 = w1/100
%w2 = w2/100

momentum1=0;
momentum2=0;
% 
size_batch=16;
js=trainsize/size_batch;
p = reshape(randperm(trainsize),[size_batch js]);
gamma1=1;
beta1=0;
gamma2=1;
beta2=0;
for iteration = 1 : max_iteration      
    for j=1:js
        rate_drop=.9;
        %
        xtemp = x(p(:,j),:);        
        %batch normalisation
        xbar = mean(xtemp,2);        
        xvar = var(xtemp')';
        xtemp = (xtemp - xbar)./sqrt(xvar+1e-8);
        xtemp1 = [ones(size_batch,1) gamma1.*xtemp+beta1];
        
        
        % calculate hidden layer
        z1 = 1 ./ (1 + exp(-w1 * xtemp1'))';
        % cauculate output layer
        %z1 = 1 ./ (1 + exp(-w2 * [ones(1,numTP); z2']))';
        drop = rand(size_batch,numHid)<rate_drop;
        z1 = z1 .* drop/rate_drop;
        z2 = w2 * [ones(1,size_batch); z1'];
        z2 = z2';
              
        z2 = softmax(z2')';
        % calculate gradient output layer
        % dz2/d(w2*z1)
        delta2 = (y(p(:,j),:) - z2);    
        % calculate gragient hidden layer
        % dz2/d(w1*xtemp1)
        delta1 = z1 .* (1 - z1) .* drop .* (delta2 * w2(:,2:end))/rate_drop;
        %delta1 = delta1.*(delta1>0);
        change2 = delta2' * [ones(size_batch,1), z1]/size_batch;
        change1 = delta1' * xtemp1/size_batch;
        % sum of training pattern
        w2_new = lr * (change2 - 0.001*w2+0.5*momentum2);
        w1_new = lr * (change1 - 0.001*w1+0.5*momentum1);
        momentum2 = change2;
        momentum1 = change1;
        dbeta = delta1 * w1(:,2:end);
        dgamma = sum(dbeta.*xtemp)/size_batch;
        dbeta = sum(dbeta)/size_batch;
        gamma1 = gamma1 + 0.0001*dgamma;
        beta1 = beta1 + 0.0001*dbeta;
        % update w2
        w2 = w2 + w2_new;
        % update w1
        w1 = w1 + w1_new;
    end    
    % mean square error
    %mse(1,iteration) = sum(sum((o-t).^2)')/(numTP*numOut)
    %[~,i]=max(a,[],2);
    %loss(iteration) = sum(i==(y(size_batch,:)*(1:10)'))/size_batch;
    %loss(iteration)
    iteration
    %% plot map and decision boundary
    % calculate hidden layer
    
    
    xbar = mean(xtest,2);        
    xvar = var(xtest')';
    xtest1 = (xtest - xbar)./sqrt(xvar+1e-8);
    xtest2 = [ones(size(xtest,1),1) gamma1.*xtest1+beta1];
    z1 = 1 ./ (1 + exp(-w1 * xtest2'))';
    % cauculate output layer
    %z1 = 1 ./ (1 + exp(-w2 * [ones(1,numTP); z2']))';
    nn = size(z1,1);
    z2 = w2 * [ones(1,nn); z1'];
    z2 = z2';
    %for i=1:9
    %    o = 1 ./ (1 + exp(-w2 * [ones(1,numTP); z']))';
    %end
    z2 = softmax(z2')';
    [~,i]=max(z2,[],2);

    accuracy = sum(i==(ytest+1)) / nn * 100
    
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
plot(1:max_iteration, loss, 'k');
hold on;
grid
title('Learning curve')
xlabel('iteration');
ylabel('accuracy')

