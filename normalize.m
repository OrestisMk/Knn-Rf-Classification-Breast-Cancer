function [Xnormalize, mu, stddev] = normalize(X)
% This function normalizes the variables to mu=0 and stddev=1.

Xnormalize = X;
mu = zeros(1, size(X, 2));
stddev = zeros(1, size(X, 2));

% normalize every column
for i=1:size(mu,2)
    mu(1,i) = mean(X(:,i)); % calculate the mean
    stddev(1,i) = std(X(:,i)); % calculate the stddev
    Xnormalize(:,i) = (X(:,i)-mu(1,i))/stddev(1,i); % subtract the mean and devide by stddev
end