close all; clear all; clc;

T2 = readtable('ML_testfile.csv');
icol = size(T2,2)
X_t_ = T2(:,1:icol-1); % predictors
y_t_ = T2(:,end); %target
X_test=X_t_{:,:}; %140 rows x 9 columns
y_test=y_t_{:,:}; %140 rows x 1 column

%% DECISION TREE BEST MODEL TESTING

dt_final_model=load('best_model_dt.mat');
dt_model=dt_final_model.dt_mdl_gridsearch;
[Yfit_dt,Sfit_dt] = predict(dt_model,X_test);
loss_dt=loss(dt_model,X_test,y_test);
loss_dt %classification error is 0.0646

diffscore = Sfit_dt(:,2)
[Xdt,Ydt,Tdt,AUCdt,OPTROCPT,suby,subnames] = perfcurve(y_test,diffscore,'4')

%Plotting ROC Curve
plot(Xdt,Ydt)
hold on
plot(OPTROCPT(1),OPTROCPT(2),'ro')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curve for Classification by Decision Tree')
hold off

%Confusion matrix and chart
m_dt=confusionmat(y_test,Yfit_dt);
m_dt
% Confusion matrix:
%     90     2
%      7    41

figure
cm_dt = confusionchart(y_test,Yfit_dt);
cm_dt

TP=m_dt(1,1); %True Positive
FP=m_dt(1,2); %False Positive
FN=m_dt(2,1); %False Negative
TN=m_dt(2,2); %True Negative

accuracy_dt = (TP+TN)/(TP+FP+TN+FN);
accuracy_dt %0.9357
precision_dt= TP/(TP+FP);
precision_dt %0.9783
recall_dt=TP/(TP+FN);
recall_dt %0.9278
f1_score_dt=(2*precision_dt*recall_dt)/(precision_dt+recall_dt);
f1_score_dt %0.9524
AUCdt %0.9632


%% RANDOM FOREST BEST MODEL TESTING

rf_final_model=load('best_model_rf.mat');
rf_model=rf_final_model.rf_mdl_gridsearch;
[Yfit_rf,Sfit_rf] = predict(rf_model,X_test);
loss_rf=loss(rf_model,X_test,y_test);
loss_rf %classification error is 0.0287

diffscore = Sfit_rf(:,2)
[Xrf,Yrf,Trf,AUCrf,OPTROCPT,suby,subnames] = perfcurve(y_test,diffscore,'4')

%Plotting ROC Curve
plot(Xrf,Yrf)
hold on
plot(OPTROCPT(1),OPTROCPT(2),'ro')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curve for Classification by Random Forest')
hold off

%Confusion matrix and chart
m_rf=confusionmat(y_test,Yfit_rf);
m_rf
%Confusion mat:
%    91     1
%     3    45

figure
cm_rf = confusionchart(y_test,Yfit_rf);
cm_rf

TP=m_rf(1,1);
FP=m_rf(1,2);
FN=m_rf(2,1);
TN=m_rf(2,2);

accuracy_rf = (TP+TN)/(TP+FP+TN+FN); 
accuracy_rf %0.9714
precision_rf= TP/(TP+FP); 
precision_rf %0.9891
recall_rf=TP/(TP+FN); 
recall_rf %0.9681
f1_score_rf=(2*precision_rf*recall_rf)/(precision_rf+recall_rf); 
f1_score_rf %0.9785
AUCrf %0.9975

%% 
%Comparison of ROC Curves of Decision Tree and Random Forest Models 
plot(Xdt,Ydt)
hold on
plot(Xrf,Yrf)
legend('Decision Tree','Random Forest','Location','Best')
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC Curves for Decision Tree and Random Forest Classification')
hold off

%% 
%Comparison of Accuracy, Precision, Recall, F1 and AUC scores of Decision
%Tree and Random Forest Models 

X = categorical({'Accuracy','Precision','Recall','F1 score','AUC'});
bar_plot=[accuracy_dt, accuracy_rf;
    precision_dt, precision_rf;
    recall_dt, recall_rf;
    f1_score_dt, f1_score_rf;
    AUCdt,AUCrf]
bar(X,bar_plot)
legend('Decision Tree','Random Forest','Location','Best')

%% Intermediate Results for Boosting(AdaBoost) Algorithm
% Since its values are mostly same with Random Forest, it is not considered to include final comparison
%

% ada_final_model=load('best_model_rf_ensemble.mat');
% ada_model=ada_final_model.rf_mdl_all_ensemble;
% [Yfit_ada,Sfit_ada] = predict(ada_model,X_test);
% loss_ada=loss(ada_model,X_test,y_test);
% loss_ada %0.0287
% 
% diffscore = Sfit_ada(:,2)
% [Xada,Yada,Tada,AUCada,OPTROCPT,suby,subnames] = perfcurve(y_test,diffscore,'4')
% 
% plot(Xada,Yada)
% hold on
% plot(OPTROCPT(1),OPTROCPT(2),'ro')
% xlabel('False positive rate') 
% ylabel('True positive rate')
% title('ROC Curve for Classification by AdaBoost')
% hold off
% 
% m_ada=confusionmat(y_test,Yfit_ada);
% m_ada
% %Confusion mat:
% %    91     1
% %     3    45
% 
% figure
% cm_ada = confusionchart(y_test,Yfit_ada);
% cm_ada
% 
% TP=m_ada(1,1);
% FP=m_ada(1,2);
% FN=m_ada(2,1);
% TN=m_ada(2,2);
% 
% accuracy_ada = (TP+TN)/(TP+FP+TN+FN); %0.9714
% accuracy_ada
% precision_ada= TP/(TP+FP); %0.9891
% precision_ada
% recall_ada=TP/(TP+FN); %0.9681
% recall_ada
% f1_score_ada=(2*precision_ada*recall_ada)/(precision_ada+recall_ada); %0.9785
% f1_score_ada
% AUCada %0.9955