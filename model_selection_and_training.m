close all; clear all; clc;

T = readtable('ML_trainfile.csv');
icol = size(T,2)
X_t = T(:,1:icol-1); % predictors
y_t = T(:,end); %target
X_train=X_t{:,:};
y_train=y_t{:,:};

%% DECISION TREE MODELLING
rng(1); % For reproducibility
dt_mdl = fitctree(X_train,y_train,'KFold',10); % Fitting decision tree model with 10 fold cross validation
view(dt_mdl.Trained{1},'Mode','graph');
classError_smpl=kfoldLoss(dt_mdl);
classError_smpl % Error is 0.0608

%Displaying decision tree
numBranches = @(x)sum(x.IsBranch);
mdl_smpl_NumSplits = cellfun(numBranches, dt_mdl.Trained); %ref. matlab website

%number of splits
figure; 
histogram(mdl_smpl_NumSplits)

%training fitting
[yfit_dt,sfit_dt] = kfoldPredict(dt_mdl);
diffscore = sfit_dt(:,2)
[xdt,ydt,tdt,AUCdtt,OPTROCPT,suby,subnames] = perfcurve(y_train,diffscore,'4')

%Plotting ROC Curve
plot(xdt,ydt)
hold on
plot(OPTROCPT(1),OPTROCPT(2),'ro')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curve for Classification by Decision Tree')
hold off

%Confusion matrix and chart
m_dt=confusionmat(y_train,yfit_dt);
m_dt
% Confusion matrix:
%     349    17
%     17   176

figure
cm_dt = confusionchart(y_train,yfit_dt);
cm_dt

TP=m_dt(1,1); %True Positive
FP=m_dt(1,2); %False Positive
FN=m_dt(2,1); %False Negative
TN=m_dt(2,2); %True Negative

accuracy_dt = (TP+TN)/(TP+FP+TN+FN);
accuracy_dt %0.9392
precision_dt= TP/(TP+FP);
precision_dt %0.9536
recall_dt=TP/(TP+FN);
recall_dt %0.9536
f1_score_dt=(2*precision_dt*recall_dt)/(precision_dt+recall_dt);
f1_score_dt %0.9536


%% Decision tree with hyperparameter optimisation(bayes opt)
rng('default') 
tallrng('default')
dt_mdl_bayesopt=fitctree(X_train,y_train,'OptimizeHyperparameters',{'MaxNumSplits', 'MinLeafSize'}, ...
    'HyperparameterOptimizationOptions',struct('Optimizer', 'bayesopt', 'MaxObjectiveEvaluations', 30, 'AcquisitionFunctionName', 'expected-improvement-plus','KFold',10));
% Fitting bayesian optimisation decision tree model with MaxNumSplits and MinleafSize hyperparameters and also 10 fold cross validation

vl_error_dt_hyp=loss(dt_mdl_bayesopt,X_train,y_train);
vl_error_dt_hyp % 0.0322

% Total elapsed time: 18.3234 seconds
% Total objective function evaluation time: 2.4568
% 
% Best observed feasible point:
%     MinLeafSize    MaxNumSplits
%     ___________    ____________
% 
%          5              43     
% 
% Observed objective function value = 0.050089
% Estimated objective function value = 0.050012
% Function evaluation time = 0.081768
% 
% Best estimated feasible point (according to models):
%     MinLeafSize    MaxNumSplits
%     ___________    ____________
% 
%          6              20     
% 
% Estimated objective function value = 0.050112
% Estimated function evaluation time = 0.070113



%% Decision tree with hyperparameter optimisation(gridsearch)
% rng('default') 
% tallrng('default')
% dt_mdl_gridsearch=fitctree(X_train,y_train,'OptimizeHyperparameters',{'MaxNumSplits', 'MinLeafSize'}, ...
%     'HyperparameterOptimizationOptions',struct('Optimizer', 'gridsearch', 'MaxObjectiveEvaluations', 30, 'AcquisitionFunctionName', 'expected-improvement-plus','KFold',10));

% Alternative to bayesian optimisation, fitting grid search decision tree model with MaxNumSplits and MinleafSize hyperparameters and also 10 fold cross validation

vl_error_dt_hyp2=loss(dt_mdl_gridsearch,X_train,y_train);
vl_error_dt_hyp2 %0.0322, same with bayesian optimisation


% Total function evaluations: 30
% Total elapsed time: 5.1583 seconds
% Total objective function evaluation time: 1.9283
% 
% Best observed feasible point:
%     MinLeafSize    MaxNumSplits
%     ___________    ____________
% 
%          3              8      
% 
% Observed objective function value = 0.042934
% Function evaluation time = 0.065551

[yfit_dt,sfit_dt] = predict(dt_mdl_gridsearch,X_train);
diffscore = sfit_dt(:,2)
[xdt,ydt,tdt,AUCdtt,OPTROCPT,suby,subnames] = perfcurve(y_train,diffscore,'1')

%Plotting ROC Curve
plot(xdt,ydt)
hold on
plot(OPTROCPT(1),OPTROCPT(2),'ro')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curve for Classification by Decision Tree')
hold off

%Confusion matrix and chart
m_dt=confusionmat(y_train,yfit_dt);
m_dt
% Confusion matrix:
%    354    12
%      6   187

figure
cm_dt = confusionchart(y_train,yfit_dt);
cm_dt

TP=m_dt(1,1); %True Positive
FP=m_dt(1,2); %False Positive
FN=m_dt(2,1); %False Negative
TN=m_dt(2,2); %True Negative

accuracy_dt = (TP+TN)/(TP+FP+TN+FN);
accuracy_dt %0.9678
precision_dt= TP/(TP+FP);
precision_dt %0.9672
recall_dt=TP/(TP+FN);
recall_dt %0.9833
f1_score_dt=(2*precision_dt*recall_dt)/(precision_dt+recall_dt);
f1_score_dt %.9752
AUCdtt %0.9813

%%Loss values are same of bayesion opt and grid search models. Cross validation loss value is better with grid search optimisation. Also gridsearch's elapsed and evaluation times are lower, means it is
%%faster. Since, we choose the gridsearch model to test for decision tree.

save('best_model_dt.mat') %saving model to use on the unseen test data

%% RANDOM FOREST MODELLING
%100 trees
rf_mdl1 = TreeBagger(100,X_train,y_train,'OOBPrediction','On','OOBPredictorImportance','on');

err = error(rf_mdl1,X_train,y_train);
err % error is 0
err2 = oobError(rf_mdl1);
err2 % out of bag error is 0.0322
yfit_rf2 = oobPredict(rf_mdl1);
[Yfit,Sfit] = oobPredict(rf_mdl1);

%Classification error vs number of grown trees
plot(err)
xlabel('Number of Grown Trees')
ylabel('Classification Error')

%oob error
b.DefaultYfit = '';
figure
plot(oobError(rf_mdl1))
xlabel('Number of Grown Trees')
ylabel('Out-of-Bag Classification Error')

%training fitting
[yfit_rf,sfit_rf] = predict(rf_mdl1,X_train);
diffscore = sfit_rf(:,2)
[xrf,yrf,trf,AUCrft,OPTROCPT,suby,subnames] = perfcurve(y_train,diffscore,'4')

%Plotting ROC Curve
plot(xrf,yrf)
hold on
plot(OPTROCPT(1),OPTROCPT(2),'ro')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curve for Classification by Random Forest')
hold off

%Confusion matrix and chart
m_rf=confusionmat(y_train,yfit_rf);
m_rf
% Confusion matrix:
%    354    12
%      6   187

figure
cm_rf = confusionchart(y_train,yfit_rf);
cm_rf

TP=m_rf(1,1); %True Positive
FP=m_rf(1,2); %False Positive
FN=m_rf(2,1); %False Negative
TN=m_rf(2,2); %True Negative

accuracy_rf = (TP+TN)/(TP+FP+TN+FN);
accuracy_rf %0.9678
precision_rf= TP/(TP+FP);
precision_rf %0.9672
recall_rf=TP/(TP+FN);
recall_rf %0.9833
f1_score_rf=(2*precision_rf*recall_rf)/(precision_rf+recall_rf);
f1_score_rf %.9752
AUCrft %0.9813


%% Random forest modeling with feature selection
%Some predictors are dropped via OOBPermutedPredictorDeltaError to see how
%the model can be more efficient
figure
bar(rf_mdl1.OOBPermutedPredictorDeltaError);
xlabel('Feature Index')
ylabel('Out-of-Bag Feature Importance')

delta=rf_mdl1.OOBPermutedPredictorDeltaError;

idxvar = find(delta>0.7);

rf_mdl1_v2 = TreeBagger(100,X_train(:,idxvar),y_train,'OOBPredictorImportance','off','OOBPrediction','on');

err_v2 = error(rf_mdl1_v2,X_train(:,idxvar),y_train);
err_v2 % error is 0

err2_v2 = oobError(rf_mdl1_v2); 
err2_v2 % out of bag error is 0.0429. It is not improved.


plot(err_v2)
xlabel('Number of Grown Trees')
ylabel('Classification Error')

b.DefaultYfit = '';
figure
plot(oobError(rf_mdl1_v2))
xlabel('Number of Grown Trees')
ylabel('Out-of-Bag Classification Error')


gPosition = find(strcmp('4',rf_mdl1_v2.ClassNames))
[fpr,tpr] = perfcurve(rf_mdl1_v2.Y,Sfit(:,gPosition),'4');
figure
plot(fpr,tpr)
xlabel('False Positive Rate')
ylabel('True Positive Rate')

[fpr,accu,thre] = perfcurve(rf_mdl1_v2.Y,Sfit(:,gPosition),'4','YCrit','Accu');
figure(20)
plot(thre,accu)
xlabel('Threshold for ''Malignant'' Returns')
ylabel('Classification Accuracy')

accuracy2=accu(abs(thre-0.5)<eps) % training accuracy is 0.9606 which is same with first RF model

%% Random forest model with hyperparameter optimisation and 10 fold cross validation
rng('default')
t = templateTree('Reproducible',true);
rf_mdl_gridsearch=fitcensemble(X_train,y_train,'OptimizeHyperparameters',{'NumLearningCycles', 'LearnRate', 'MaxNumSplits', 'MinLeafSize'}, ...
    'Learners',t,'HyperparameterOptimizationOptions',struct('Optimizer', 'gridsearch', 'MaxObjectiveEvaluations', 30, ...
    'AcquisitionFunctionName', 'expected-improvement-plus','kFold',10));

loss_rf_grid=loss(rf_mdl_gridsearch,X_train,y_train)
loss_rf_grid % loss is 0.0161 which is very good


% MaxObjectiveEvaluations of 30 reached.
% Total function evaluations: 30
% Total elapsed time: 150.0704 seconds
% Total objective function evaluation time: 146.1429
% 
% Best observed feasible point:
%     NumLearningCycles    LearnRate    MinLeafSize    MaxNumSplits
%     _________________    _________    ___________    ____________
% 
%            210           0.046416         12              8      
% 
% Observed objective function value = 0.039356
% Function evaluation time = 8.0002

%training fitting
[yfit_rf,sfit_rf] = predict(rf_mdl_gridsearch,X_train);
diffscore = sfit_rf(:,2)
[xrf,yrf,trf,AUCrft,OPTROCPT,suby,subnames] = perfcurve(y_train,diffscore,'4')

%Plotting ROC Curve
plot(xrf,yrf)
hold on
plot(OPTROCPT(1),OPTROCPT(2),'ro')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curve for Classification by Random Forest')
hold off

%Confusion matrix and chart
m_rf=confusionmat(y_train,yfit_rf);
m_rf
% Confusion matrix:
%    359     7
%      2   191

figure
cm_rf = confusionchart(y_train,yfit_rf);
cm_rf

TP=m_rf(1,1); %True Positive
FP=m_rf(1,2); %False Positive
FN=m_rf(2,1); %False Negative
TN=m_rf(2,2); %True Negative

accuracy_rf = (TP+TN)/(TP+FP+TN+FN);
accuracy_rf %0.9839
precision_rf= TP/(TP+FP);
precision_rf %0.9809
recall_rf=TP/(TP+FN);
recall_rf %0.9945
f1_score_rf=(2*precision_rf*recall_rf)/(precision_rf+recall_rf);
f1_score_rf %0.9876
AUCrft %0.9994


save('best_model_rf.mat')

%% Rest ensemble selection scenerio(Boosting)
rng('default')
t = templateTree('Reproducible',true);
mdl_all_ensemble=fitcensemble(X_train,y_train,'OptimizeHyperparameters', 'all');

loss_ensemble=loss(mdl_all_ensemble,X_train,y_train)
loss_ensemble % loss is 0


% MaxObjectiveEvaluations of 30 reached.
% Total function evaluations: 30
% Total elapsed time: 43.2695 seconds
% Total objective function evaluation time: 32.4009
% 
% Best observed feasible point:
%       Method      NumLearningCycles    LearnRate    MinLeafSize    MaxNumSplits    SplitCriterion    NumVariablesToSample
%     __________    _________________    _________    ___________    ____________    ______________    ____________________
% 
%     AdaBoostM1           62             0.7137           2             136              gdi                  NaN         
% 
% Observed objective function value = 0.0322
% Estimated objective function value = 0.03222
% Function evaluation time = 1.4251
% 
% Best estimated feasible point (according to models):
%       Method      NumLearningCycles    LearnRate    MinLeafSize    MaxNumSplits    SplitCriterion    NumVariablesToSample
%     __________    _________________    _________    ___________    ____________    ______________    ____________________
% 
%     AdaBoostM1           83             0.87183          2             234              gdi                  NaN         
% 
% Estimated objective function value = 0.03219
% Estimated function evaluation time = 1.7225

%training fitting
[yfit_rf,sfit_rf] = predict(mdl_all_ensemble,X_train);
diffscore = sfit_rf(:,2)
[xrf,yrf,trf,AUCrft,OPTROCPT,suby,subnames] = perfcurve(y_train,diffscore,'4')

%Plotting ROC Curve
plot(xrf,yrf)
hold on
plot(OPTROCPT(1),OPTROCPT(2),'ro')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curve for Classification by Random Forest')
hold off

%Confusion matrix and chart
m_rf=confusionmat(y_train,yfit_rf);
m_rf
% Confusion matrix:
%    366     0
%      0   193

figure
cm_rf = confusionchart(y_train,yfit_rf);
cm_rf

TP=m_rf(1,1); %True Positive
FP=m_rf(1,2); %False Positive
FN=m_rf(2,1); %False Negative
TN=m_rf(2,2); %True Negative

accuracy_rf = (TP+TN)/(TP+FP+TN+FN);
accuracy_rf %1
precision_rf= TP/(TP+FP);
precision_rf %1
recall_rf=TP/(TP+FN);
recall_rf %1
f1_score_rf=(2*precision_rf*recall_rf)/(precision_rf+recall_rf);
f1_score_rf %1
AUCrft %1


save('best_model_rf_ensemble.mat')
