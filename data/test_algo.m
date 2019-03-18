%use toolbox
xx=x';
yy=y';
net = trainSoftmaxLayer(xx,yy);
yyy=net(x');
[~,i]=max(yyy,[],1);
accuracy2 = sum(i==(1:10)*yy) / 50000 * 100
%0.74
yyy=net(xtest');
[~,i]=max(yyy,[],1);
accuracy2 = sum(i==(ytest+1)') / 10000 * 100
t = templateLinear('Learner','SVM','Regularization','Ridge','Lambda',0:0.1:1)
t = templateSVM('KernelFunction','gaussian')
Mdl = fitcecoc(x,y*(1:10)','Learners',t)
sum(Mdl.predict(x)==y*(1:10)')/50000
sum(Mdl.predict(xtest)==(ytest+1))/10000
[B,dev,stats] = mnrfit(x,y*(1:10)')
tTree = templateTree('surrogate','on');
tEnsemble = templateEnsemble('GentleBoost',8,tTree);
pool = parpool; % Invoke workers
options = statset('UseParallel',true);
Mdl2 = fitcecoc(x,y*(1:10)','Coding','onevsall','Learners',tEnsemble,...
                'Prior','uniform','Options',options);
sum(Mdl2.predict(x)==y*(1:10)')/50000
sum(Mdl2.predict(xtest)==(ytest+1))/10000