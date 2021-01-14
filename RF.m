tic
clear;
clc;

%import data

brc = readtable('Breast_cancer2.csv');

% description of brc reveals the existence of three columns with
% significant outliers (area_mean , area_se , area_worst)
summary(brc);

% size table (569 rows, 32 columns)
[r c] = size(brc);

% seperation of dependent and independent variables
X = table2array(brc(:,3:32));
Y = table2array(brc(:,2));

%save all variables
dep_variables = brc.Properties.VariableNames;
X_variables = brc(:,3:end);



%define classes
count_y0_y1 = tabulate(Y)% we have 357 malignant and 212 benign tumor

%seperate data into training and test sets using cv partition
%for slight imbalance in target class
cv = cvpartition(Y,'holdout',0.3);
X_Train = X(training(cv,1),:);
y_Train = Y(training(cv,1));
X_Test = X(test(cv,1),:);
y_Test = Y(test(cv,1),:);

% Data normalization
%We applied data normalization with mu=0 and standard deviation = 1 to the
%variables in response to 3 signigicant outliers in three from
%the summary.


[X_Train, mu, stddev] = normalize(X_Train);

for i=1:size(X_Test, 2)
    X_Test(:,i) = (X_Test(:,i)-mu(1,i))/stddev(1,i);
end
%feature scaling is restricted to the test set.

%We opted for Bayesian optimisation for the first tuning of
%hyperparameter because of  highly correlated dataset.
% This method create one figure(beacause we optimize 3 variables) : 
%a) second shows the estimated and observed value of the Objective function

%Three variables are created with spesific name,Type and range 

feat_2 = optimizableVariable('feat_2',[1, 50],'Type','integer');
split_2 = optimizableVariable('split_2',{'gdi', 'deviance'}, 'Type','categorical');
num_lc2 = optimizableVariable('num_lc2',[10, 400], 'Type','integer');
n2 = size(X_Train, 1);
rng(1);
cv2 = cvpartition(n2,'Kfold',10);

%Objective function, Returns a measure of loss for hyperparameter tuning
fun = @(x)kfoldLoss(fitcensemble(X_Train, y_Train, 'CVPartition', cv2, 'Method', 'Bag'));

% set the limit for objective evaluation than acquisition function and expected improvement plus after trying for 150 observations
bo_res2 = bayesopt(fun,[feat_2, split_2, num_lc2],'Verbose',1, 'MaxObjectiveEvaluations',150)

%saved best combination which give min_error
feat2_bo = bo_res2.XAtMinObjective.feat_2;
split2_bo = bo_res2.XAtMinObjective.split_2;
numlc2_bo = bo_res2.XAtMinObjective.num_lc2;




% minimun error for the set of hyperparameters that evaluated on validation set
min_error =bo_res2.MinObjective;


% Create optimised tree and generate results
t2 = templateTree('NumVariablesToSample', feat2_bo, 'SplitCriterion', char(split2_bo));


% and refit the RF with this tree template as the base learner
rng(1)
bomdl2 = fitcensemble(X_Train, y_Train, 'Method', 'Bag', 'NumLearningCycles',numlc2_bo,'learners', t2);
bomdl2_loss = loss(bomdl2,X_Test, y_Test); %classification effective of training data based on model predictions
bomdl2_rloss = resubLoss(bomdl2); %Misclassifications from the predictions above 


% Create Matrix for bayesian optimisation results and test model
figure()
bo_Predict2 = predict(bomdl2, X_Test);
cm_bo_rf = confusionmat(bo_Predict2, y_Test)
matrx = confusionchart(bo_Predict2, y_Test)

%accursacy
Accuracy_bo = 100*(cm_bo_rf(1,1)+cm_bo_rf(2,2))./(cm_bo_rf(1,1)+cm_bo_rf(2,2)+cm_bo_rf(1,2)+cm_bo_rf(2,1))

%accurcy of the model with bayesopt tuning
rf_bo_precision = cm_bo_rf(1,1)./(cm_bo_rf(1,1)+cm_bo_rf(1,2));

%Recall is the ability to identify the number of samples that would really count positive for tumor.
rf_bo_recall = cm_bo_rf(1,1)./(cm_bo_rf(1,1)+cm_bo_rf(2,1));

%F1-score means a statistical measure of the accuracy. Also , F1 score is used because FN and TN are crusial for our results.
f1_Scores_bo = 2*(rf_bo_precision.*rf_bo_recall)./(rf_bo_precision+rf_bo_recall)

% Identify misclassified tumours
testheight = size(X,1)
trainheight = size(Y,1)
misClass_rf_bayesopt = (cm_bo_rf(1,2)+cm_bo_rf(2,1));
errTumour_bo = 100*misClass_rf_bayesopt/testheight;



%Grid search cross validation for HP tuning. 
%This method is highly computationally expensive,
%we have reduced steps - the biggest run was done  
%at 2000 trees but we have set this at 100 for computational efficiency.

numTrees = 100

%Create arrays to house hyperparameter performance, based on error
% and average error create a boolean that will
%track hyperparameter performance

opt_LS_grdSrch= 100
opt_PTS_grdSrch = 100
base_misclass = 1
for min_LS = 1:20
    for num_PTS = 1:30
        rf_mdl_grdSrch = TreeBagger(numTrees, X_Train, y_Train, 'Method', 'classification', 'OOBPrediction', 'on', 'MinLeafSize', min_LS, 'NumPredictorsToSample',num_PTS);
        misclass_grdSrch = oobError(rf_mdl_grdSrch);
        avg_misclass =sum(misclass_grdSrch)/numTrees
        if avg_misclass < base_misclass
            opt_LS_grdSrch= min_LS;
            opt_PTS_grdSrch = num_PTS;
            base_misclass = avg_misclass;
        end
    end
end

%create optimised model based on results of hyperparameter tuning
bg_Mdl_grdSrch = TreeBagger(numTrees, X_Train, y_Train, 'Method', 'Classification', 'OOBPrediction', 'on', 'MinLeafSize', opt_LS_grdSrch, 'NumPredictorsToSample',opt_PTS_grdSrch)

%Print best hyperparameters
opt_LS_grdSrch
opt_PTS_grdSrch

%test model
label1 = predict(bg_Mdl_grdSrch, X_Test)
label2 = str2double(label1)
ta = table(y_Test, label1, 'VariableNames', {'TrueLabel', 'PredictedLabel'});

%create confusion matrix
figure()
rf_cnf_grdSrch = confusionmat(label2, y_Test)
rf_con_chart = confusionchart(label2, y_Test)

%accurcy of the model with bayesopt tuning
Accuracy_rf_grdSrch = 100*(rf_cnf_grdSrch(1,1)+rf_cnf_grdSrch(2,2))./(rf_cnf_grdSrch(1,1)+rf_cnf_grdSrch(2,2)+rf_cnf_grdSrch(1,2)+rf_cnf_grdSrch(2,1))

%Precision defines the accuracy of judgment.
rf_precision_grdSrch = rf_cnf_grdSrch(1,1)./(rf_cnf_grdSrch(1,1)+rf_cnf_grdSrch(1,2));

%Recall is the ability to identify the number of samples that would really count positive for tumor.
rf_recall_grdSrch = rf_cnf_grdSrch(1,1)./(rf_cnf_grdSrch(1,1)+rf_cnf_grdSrch(2,1));

%F1 score means a statistical measure of the accuracy
f1_Scores_grdSrch = 2*(rf_precision_grdSrch.*rf_recall_grdSrch)./(rf_precision_grdSrch+rf_recall_grdSrch)

% Identify misclassified tumours
testheight = size(X,1)
trainheight = size(Y,1)
misClass_rf_grdSrch = (rf_cnf_grdSrch(1,2)+rf_cnf_grdSrch(2,1));
errTumour_grdSrch = 100*misClass_rf_grdSrch/testheight;

toc
