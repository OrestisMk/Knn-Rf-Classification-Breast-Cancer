tic
clear;
clc;

%import data

ctg = readtable('ctg_datapreproccecing.csv');

% a description of brc, reveals the existance of three columns with
% significant outliers (area_mean , area_se , area_worst)
summary(ctg);

% size table (569 rows, 32 columns)
[r c] = size(ctg);

% seperation of depentent and independent variables
X = table2array(brc(:,3:32));
Y = table2array(brc(:,2));

%save all variables
dep_variables = brc.Properties.VariableNames;
X_variables = brc(:,3:end);


 
 %define classes
count_y0_y1 = tabulate(Y)% we have 357 malignant and 212 benign tumor
 


%seperate data into training and test sets using cv partition to account
%for slight imbalance in target class
cv = cvpartition(Y,'holdout',0.3);
X_Train = X(training(cv,1),:);
y_Train = Y(training(cv,1));
X_Test = X(test(cv,1),:);
y_Test = Y(test(cv,1),:);




% Data normalization
%We applied data normalization with mu=0 and standard deviation = 1 to the
%variables. This was response to 3 signigicant outliers in two colums from
%the summary.This step is critical for KNN because it is very sensitive
%with outlier data points.

[X_Train, mu, stddev] = normalize(X_Train); 


for i=1:size(X_Test, 2)
    X_Test(:,i) = (X_Test(:,i)-mu(1,i))/stddev(1,i);
end
 % feature scaling which is restricted to the test set.


%We opted for Bayesian optimisation for the first tuning of
%hyperparameter because of  highly correlated dataset
% This method create to figures : 
%a) first one illustrates the two HP and the corresponding value of each repetition
%b) second shows the estimated and observed value of the Objective function

%Two variables are created with spesific name,Type and range 
num_k = optimizableVariable('num_k',[1,50],'Type','integer');

dst = optimizableVariable('dst', {'minkowski','correlation','hamming',...
   'jaccard','mahalanobis','cityblock','euclidean','cosine','spearman'...
    'seuclidean','chebychev'},'Type','categorical');
rng(1)

%Index and partition for second cvpartitioned dataset
foldIdx = size(X_Train,1)
cv1 = cvpartition(foldIdx, 'kfold', 10)

%Objective function, Returns a measure of loss for hyperparameter tuning
fun = @(x)kfoldLoss(fitcknn(X_Train,y_Train,'CVPartition',cv1,'NumNeighbors', x.num_k,'Distance',char(x.dst), 'NSMethod','exhaustive'));

% set the limit for objective evaluation (for formulating acquisition function and expected improvement plus) after trying for 200 observations
results_bayesopt = bayesopt(fun,[num_k,dst], 'Verbose',1,...
    'MaxObjectiveEvaluations', 200);

%saved best combination which give min_error
num_k_bayesopt = results_bayesopt.XAtMinObjective.num_k;
dst2 = results_bayesopt.XAtMinObjective.dst;


% minimun error for the set of hyperparameters that evaluated on validation set
min_error_bayesopt  = results_bayesopt.MinObjective; 



%Fit optimised hyper parameter model
rng(1)
knn_bayesopt = fitcknn(X_Train , y_Train, 'NumNeighbors', num_k_bayesopt,'Distance',char(dst2));
[knn_yPrd_bayesopt, knn_scr_bayesopt,conio] = predict(knn_bayesopt,X_Test);% returns predicted class labels based on the trained classification model
knn_loss = loss(knn_bayesopt,X_Test, y_Test); %classification effective of training data based on model predictions
knn_rloss = resubLoss(knn_bayesopt); %Misclassifications from the predictions above 




%%% confusion matrix
figure()


[knn_bayesopt_cm, order] = confusionmat(y_Test, knn_yPrd_bayesopt)
cm1chart_bayesopt = confusionchart( y_Test, knn_yPrd_bayesopt)


%accurcy of knn model with bayesopt tuning.
Accuracy_knn_bayesopt = 100*(knn_bayesopt_cm(1,1)+knn_bayesopt_cm(2,2))./(knn_bayesopt_cm(1,1)+knn_bayesopt_cm(2,2)+knn_bayesopt_cm(1,2)+knn_bayesopt_cm(2,1))

%Precision defines the accuracy of judgment.
knn_precision = knn_bayesopt_cm(1,1)./(knn_bayesopt_cm(1,1)+knn_bayesopt_cm(1,2));


%Recall is the ability to identify the number of samples that would really count positive for tumor.
knn_recall = knn_bayesopt_cm(1,1)./(knn_bayesopt_cm(1,1)+knn_bayesopt_cm(2,1));

%F1-score means a statistical measure of the accuracy. Also , F1 score is used because FN and TN are crusial for our results. 
f1_Scores_bayesopt = 2*(knn_precision.*knn_recall)./(knn_precision+knn_recall)




% Identify misclassified tumours
testheight = size(X,1)
trainheight = size(Y,1)
misClass_bayesopt = (knn_bayesopt_cm(1,2)+knn_bayesopt_cm(2,1));
errTumour = 100*misClass_bayesopt/testheight;


%We use grid search as an alternative method for HP tuning parameters.
%Also,  the same range for the HP is used.
%This method is highly computationally expensive. 
    
%ignore error and loss
min_error = 1 
grdSrch_loss = []

    
    for num_dst = 1:11
        optional_dst = ["minkowski","correlation","hamming",...
    "jaccard","mahalanobis","cityblock","euclidean","cosine","spearman"...
    "seuclidean","chebychev"]

        for num_k2 = 1:50
        
            test_dist = optional_dst(num_dst);
            Mdl_grdSrch = fitcknn(X_Train, y_Train, 'NumNeighbors',num_k2, 'Distance', test_dist);
            cv_Mdl_grdSrch = crossval(Mdl_grdSrch);
            grdSrch_kloss=kfoldLoss(cv_Mdl_grdSrch);
        
            grdSrch_loss=[grdSrch_loss grdSrch_kloss]
        
        %  save optimum num_k and dist if it finds a better min_error than
        %  kloss
            if grdSrch_kloss<min_error;
                min_error=grdSrch_kloss;
                optimum_k=num_k2;
                optimum_dst=test_dist;
            end
        end
    end
    
    optimum_k
    optimum_dst
    
%Different results for number of neighbors and distance from Bayes opt.

%train model with the optimum parametres from grid

knn_grdSrch = fitcknn(X_Train, y_Train, 'NumNeighbors',optimum_k,'Distance',optimum_dst);

% test mdl

[knn_yPrd_grdSrch, knn_scr_grdSrch,conio2] = predict(knn_grdSrch, X_Test);
figure() 
[knn_grdSrc_cm, order] = confusionmat(y_Test,knn_yPrd_grdSrch);
cm2chart_grdSrch = confusionchart( y_Test, knn_yPrd_grdSrch)
cm.title = 'KNN_with_grdSrch'

knn_testErr_grdSrch = loss(knn_grdSrch, X_Test, y_Test);
errTrain = resubLoss(knn_grdSrch); 
  optimum_k
  optimum_dst
  
  
%accuracy of grid search
Accuracy_grdSrch = 100*(knn_grdSrc_cm(1,1)+knn_grdSrc_cm(2,2))./(knn_grdSrc_cm(1,1)+knn_grdSrc_cm(2,2)+knn_grdSrc_cm(1,2)+knn_grdSrc_cm(2,1))% what is this?

misClass_grdSrch = (knn_grdSrc_cm(1,2)+knn_grdSrc_cm(2,1));
% miss classification
errTumour_grdSrch = 100*misClass_grdSrch/testheight;

% F1 score is used because FN and TN are crusial for our results. 
knn_precision_grdSrch = knn_grdSrc_cm(1,1)./(knn_grdSrc_cm(1,1)+knn_grdSrc_cm(1,2));
knn_recall_grdSrch = knn_grdSrc_cm(1,1)./(knn_grdSrc_cm(1,1)+knn_grdSrc_cm(2,1));
f1_Scores_grdSrch = 2*(knn_precision_grdSrch.*knn_recall_grdSrch)./(knn_recall_grdSrch+knn_recall_grdSrch)

toc
