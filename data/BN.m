% chen chen 03/18/2019
% backpropagation(4 layeres) without neural networ toolbox
% accuracy 86%
% perhaps need to initialise a few times to see whether the starting point
% is within the range.

clear, close all


%
%load inputdata
test_128=hdf5info('test_128.h5');
test= hdf5read(test_128.GroupHierarchy.Datasets)';
train_128=hdf5info('train_128.h5');
x= hdf5read(train_128.GroupHierarchy.Datasets)';
train_label=hdf5info('train_label.h5');
y= double(hdf5read(train_label.GroupHierarchy.Datasets));



% ming test
% test = rand(10000, 128);
% x = rand(60000, 128);
% y = randi([0, 9], 60000, 1);

trainsize=50176;

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
max_iteration = 250;
for hidden_layer_dim= 161
%numHid = 190; % hidden(midle) layer's unit size

% init
loss = zeros(1 , max_iteration);
w1_new = zeros(hidden_layer_dim, numFV + 1);
w2_new = zeros(hidden_layer_dim, hidden_layer_dim + 1);
w3_new = zeros(numOut, hidden_layer_dim + 1);

% weight value range[-1-1]
rng(3)
w1 = 2 * rand(numFV + 1,hidden_layer_dim)' - 1;
w2 = 2 * rand(hidden_layer_dim + 1,hidden_layer_dim)' - 1;
w3 = 2 * rand(hidden_layer_dim + 1,numOut)' - 1;
%w1 = w1/100
%w2 = w2/100

momentum1=0;
momentum2=0;
momentum3=0;
% 
size_batch=1024;
batches=trainsize/size_batch;
p = reshape(1:trainsize,[size_batch batches]);
gamma1=ones(1,numFV);
beta1=zeros(1,numFV);
gamma2=ones(1,hidden_layer_dim);
beta2=zeros(1,hidden_layer_dim);
means1=zeros(batches,numFV);
vars1=zeros(batches,numFV);
means2=zeros(batches,hidden_layer_dim);
vars2=zeros(batches,hidden_layer_dim);
for iteration = 1 : max_iteration
    
    for j=1:batches
        rate_drop=1;
        %
        xtemp = x(p(:,j),:);        
        %batch normalisation
        xbar = mean(xtemp,1);        
        xvar = var(xtemp);
        means1(j,:)=xbar;
        vars1(j,:)=xvar;
        xtemp = (xtemp - xbar)./sqrt(xvar+1e-8);
        xtemp1 = [ones(size_batch,1) gamma1.*xtemp+beta1];
        
        
        % calculate hidden layer
        z1 = 1 ./ (1 + exp(-w1 * xtemp1'))';
        % cauculate output layer
        %z1 = 1 ./ (1 + exp(-w2 * [ones(1,numTP); z2']))';
        rng(3)
        drop = rand(hidden_layer_dim,size_batch)'<rate_drop;
        z1 = z1 .* drop/rate_drop;
        
        %zbar = mean(z1,2);        
        %zvar = var(z1')';
        %zbar = (zbar - zbar)./sqrt(zvar+1e-8);
        zbar = mean(z1,1);        
        zvar = var(z1);
        means2(j,:)=zbar;
        vars2(j,:)=zvar;
        ztemp = (z1 - zbar)./sqrt(zvar+1e-8);                
        ztemp1 = [ones(size_batch,1) gamma2.*ztemp+beta2]; 
        
        z2 = w2 * ztemp1';
        z2 = z2 .* (z2>0);      
        z2 = z2';
        
        z3 = w3 * [ones(1,size_batch); z2'];
        z3 = z3';              
        z3 = softmax(z3')';
        % calculate gradient output layer
        % dz2/d(w2*z1)
        delta3 = (y(p(:,j),:) - z3);    
        % calculate gragient hidden layer
        delta2 = delta3 * w3(:,2:end).*(z2>0);
        % calculate gragient hidden layer
        % dz2/d(w1*xtemp1)
        delta1 = z1 .* (1 - z1) .* drop .* (delta2 * w2(:,2:end))/rate_drop;
        %delta1 = delta1.*(delta1>0);
        change3 = delta3' * [ones(size_batch,1), z2]/size_batch;
        %change2 = delta2' * [ones(size_batch,1), z1]/size_batch;
        change2 = delta2' * ztemp1/size_batch;
        change1 = delta1' * xtemp1/size_batch;
        % sum of training pattern
        mm=0.95-0.005*min(iteration,15);
        w3_new = lr * (change3 - 0.0005*w3)+mm*momentum3;
        w2_new = lr * (change2 - 0.0005*w2)+mm*momentum2;
        w1_new = lr * (change1 - 0.0005*w1)+mm*momentum1;
        momentum3 = w3_new;
        momentum2 = w2_new;
        momentum1 = w1_new;
        dbeta = delta1 * w1(:,2:end);
        dgamma = sum(dbeta.*xtemp)/size_batch;
        dbeta = sum(dbeta)/size_batch;
        gamma1 = gamma1 + lr*dgamma;
        beta1 = beta1 + lr*dbeta;
        
        dbeta = delta2 * w2(:,2:end);
        dgamma = sum(dbeta.*ztemp)/size_batch;
        dbeta = sum(dbeta)/size_batch;
        gamma2 = gamma2 + lr*dgamma;
        beta2 = beta2 + lr*dbeta;
        % update w2
        w3 = w3 + w3_new;
        w2 = w2 + w2_new;
        % update w1
        w1 = w1 + w1_new;
    end    
    % mean square error
    %mse(1,iteration) = sum(sum((o-t).^2)')/(numTP*numOut)
    %[~,i]=max(a,[],2);
    %
    %loss(iteration)
    %iteration
    %% plot map and decision boundary
    % calculate hidden layer
        

    
    xbar = mean(means1,1);
    xvar = mean(vars1,1)*size_batch/(size_batch-1);
    xtest1 = (xtest - xbar)./sqrt(xvar+1e-8);
    xtest2 = [ones(size(xtest,1),1) gamma1.*xtest1+beta1];
    
    z1 = 1 ./ (1 + exp(-w1 * xtest2'))';    
    zbar = mean(means2,1);        
    zvar = mean(vars2,1)*size_batch/(size_batch-1);
    ztemp = (z1 - zbar)./sqrt(zvar+1e-8);      
    z11 = [ones(size(xtest,1),1) gamma2.*ztemp+beta2]; 
    
    z2 = w2 * z11';
    z2 = z2 .* (z2>0);      
    z2 = z2';
    % cauculate output layer
    %z1 = 1 ./ (1 + exp(-w2 * [ones(1,numTP); z2']))';
    nn = size(z1,1);
    z3 = w3 * [ones(1,nn); z2'];
    z3 = z3';
    %for i=1:9
    %    o = 1 ./ (1 + exp(-w2 * [ones(1,numTP); z']))';
    %end
    z3 = softmax(z3')';
    [~,i]=max(z3,[],2);

    accuracy = sum(i==(ytest+1)) / nn * 100;    
    if iteration==100 |iteration==200 |iteration==250|1
        hidden_layer_dim
        iteration
        accuracy
        %max(loss(1:iteration))
    end
    loss(iteration) = accuracy;
    
end
end
% visualize learning
figure(2)
subplot(4,1,1)
imagesc(w1(:,2:end)', [min(min(w1(:,2:end))), max(max(w1(:,2:end)))]);
title('input-hidden weight')
ylabel('input layer')
xlabel('hidden layer')
ax = gca;
ax.XTick = 1:3:hidden_layer_dim;
ax.YTick = 1:20:numFV;
colorbar

subplot(4,1,2)
imagesc(w2(:,2:end), [min(min(w2(:,2:end))), max(max(w2(:,2:end)))]);
title('hidden-hidden weight')
xlabel('hidden layer')
ylabel('hidden layer')
ax = gca;
ax.XTick = 1:3:hidden_layer_dim;
ax.YTick = 1:5:hidden_layer_dim;
colormap(hot);
colorbar

subplot(4,1,3)
imagesc(w3(:,2:end), [min(min(w3(:,2:end))), max(max(w3(:,2:end)))]);
title('hidden-output weight')
xlabel('hidden layer')
ylabel('output layer')
ax = gca;
ax.XTick = 1:3:hidden_layer_dim;
ax.YTick = 1:5:numOut;
colormap(hot);
colorbar

%
subplot(4,1,4)
plot(1:max_iteration, loss, 'k');
hold on;
grid
title('Learning curve')
xlabel('iteration');
ylabel('accuracy')

print -depsc2
eps2pdf('figure2.eps')