clear all;
close all;

Part1 = 1;
Part2 = 1;
dimension = 2;

% Defining different sample sizes
D.d100.N = 20;
D.d1000.N = 200;
D.d10k.N = 2000;
D.d20k.N = 10000;
DType = fieldnames(D);

% Defining parameters for GMM
p = [0.6, 0.4];

mu0 = [-1, -1; 1, 1]';
Sigma0(:,:,1) = [1, 0; 0, 1];
Sigma0(:,:,2) = [1, 0; 0, 1];
alpha0 = [0.5, 0.5];

mu1 = [-1, 1; 1, -1]';
Sigma1(:,:,1) = [2, 0; 0, 2];
Sigma1(:,:,2) = [2, 0; 0, 2];
alpha1 = [0.5, 0.5];

figure;

% Generating data points based on GMM, assigning labels for classes plotting the data points in different colors for each class
for ind = 1:length(DType)
    D.(DType{ind}).x = zeros(dimension, D.(DType{ind}).N);

    D.(DType{ind}).labels = rand(1, D.(DType{ind}).N) >= p(1);
    D.(DType{ind}).N0 = sum(~D.(DType{ind}).labels);
    D.(DType{ind}).N1 = sum(D.(DType{ind}).labels);
    D.(DType{ind}).phat(1) = D.(DType{ind}).N0 / D.(DType{ind}).N;
    D.(DType{ind}).phat(2) = D.(DType{ind}).N1 / D.(DType{ind}).N;

    [D.(DType{ind}).x(:, ~D.(DType{ind}).labels), ...
     D.(DType{ind}).dist(:, ~D.(DType{ind}).labels)] = ...
        randGMM(D.(DType{ind}).N0, alpha0, mu0, Sigma0);
    
    [D.(DType{ind}).x(:, D.(DType{ind}).labels), ...
     D.(DType{ind}).dist(:, D.(DType{ind}).labels)] = ...
        randGMM(D.(DType{ind}).N1, alpha1, mu1, Sigma1);
    
    subplot(2, 2, ind);
    plot(D.(DType{ind}).x(1, ~D.(DType{ind}).labels), ...
         D.(DType{ind}).x(2, ~D.(DType{ind}).labels), 'r.', 'DisplayName', 'Class 0');
    hold all;
    plot(D.(DType{ind}).x(1, D.(DType{ind}).labels), ...
         D.(DType{ind}).x(2, D.(DType{ind}).labels), 'c.', 'DisplayName', 'Class 1');
    grid on;
    xlabel('x1');
    ylabel('x2');
    title([num2str(D.(DType{ind}).N) ' Samples Distribution']);
end

legend 'show';

% Evaluating GMMs for a specific sample size, Calculating discriminant scores, probabilities, & error metrics, Plotting ROC curve, minimum error point, and classifier decisions 
if Part1
    px0 = evalGMM(D.d20k.x, alpha0, mu0, Sigma0);
    px1 = evalGMM(D.d20k.x, alpha1, mu1, Sigma1);
    discScore = log(px1 ./ px0);
    sortDS = sort(discScore);

    logGamma = [min(discScore) - eps, sort(discScore) + eps];
    prob = CalcProb(discScore, logGamma, D.d20k.labels, D.d20k.N0, D.d20k.N1, D.d20k.phat);
    logGamma_ideal = log(p(1) / p(2));
    decision_ideal = discScore > logGamma_ideal;
    p10_ideal = sum(decision_ideal == 1 & D.d20k.labels == 0) / D.d20k.N0;
    p11_ideal = sum(decision_ideal == 1 & D.d20k.labels == 1) / D.d20k.N1;
    pFE_ideal = (p10_ideal * D.d20k.N0 + (1 - p11_ideal) * D.d20k.N1) / (D.d20k.N0 + D.d20k.N1);

    [prob.min_pFE, prob.min_pFE_ind] = min(prob.pFE);
    if length(prob.min_pFE_ind) > 1
        [~, minDistTheory_ind] = min(abs(logGamma(prob.min_pFE_ind) - logGamma_ideal));
        prob.min_pFE_ind = prob.min_pFE_ind(minDistTheory_ind);
    end

    minGAMMA = exp(logGamma(prob.min_pFE_ind));
    prob.min_FP = prob.p10(prob.min_pFE_ind);
    prob.min_TP = prob.p11(prob.min_pFE_ind);

    plotROC(prob.p10, prob.p11, prob.min_FP, prob.min_TP);
    hold all;
    plot(p10_ideal, p11_ideal, 'x', 'DisplayName', 'Ideal Min. Error');
    plotMinPFE(logGamma, prob.pFE, prob.min_pFE_ind);
    plotDecisions(D.d20k.x, D.d20k.labels, decision_ideal);

    plotERMContours(D.d20k.x, alpha0, mu0, Sigma0, alpha1, mu1, Sigma1, logGamma_ideal);
end
 
% Part 2: Classification with MLP Estimation 

options = optimset('MaxFunEvals', 60000, 'MaxIter', 20000);

% Performing MLP Estimation for linear and quadratic logistic fits, calculating decision scores, probabilities, and classifier decisions 
% Plot data points with classifier decisions for both linear and quadratic fits
for ind = 1:length(DType)
    lin.x = [ones(1, D.(DType{ind}).N); D.(DType{ind}).x];
    lin.init = zeros(dimension + 1, 1);

    lin.theta = fminsearch(@(theta)(costFun(theta, lin.x, D.(DType{ind}).labels)), lin.init, options);
    lin.discScore = lin.theta' * [ones(1, D.d20k.N); D.d20k.x];
    gamma = 0;
    lin.prob = CalcProb(lin.discScore, gamma, D.d20k.labels, D.d20k.N0, D.d20k.N1, D.d20k.phat);

    quad.x = [ones(1, D.(DType{ind}).N); D.(DType{ind}).x;...
              D.(DType{ind}).x(1, :).^2;...
              D.(DType{ind}).x(1, :).*D.(DType{ind}).x(2, :);...
              D.(DType{ind}).x(2, :).^2];
    quad.init = zeros(2*(dimension + 1), 1);

    quad.theta = fminsearch(@(theta)(costFun(theta, quad.x, D.(DType{ind}).labels)), quad.init, options);
    quad.xScore = [ones(1, D.d20k.N); D.d20k.x; D.d20k.x(1, :).^2;...
                   D.d20k.x(1, :).*D.d20k.x(2, :); D.d20k.x(2, :).^2];
    quad.discScore = quad.theta' * quad.xScore;
    gamma = 0;
    quad.prob = CalcProb(quad.discScore, gamma, D.d20k.labels, D.d20k.N0, D.d20k.N1, D.d20k.phat);

    plotDecisions(D.d20k.x, D.d20k.labels, lin.prob.decisions);
    title(sprintf('Data & Classifier Decisions Against True Label for Linear Logistic Fit\nProbability of Error=%1.1f%%', 100*lin.prob.pFE));
    
    plotDecisions(D.d20k.x, D.d20k.labels, quad.prob.decisions);
    title(sprintf('Data & Classifier Decisions Against True Label for Quadratic Logistic Fit\nProbability of Error=%1.1f%%', 100*quad.prob.pFE));
end

% Computing the cost function for logistic regression
function cost = costFun(theta, x, labels)
    h = 1./(1 + exp(-x' * theta));
    cost = -1/length(h) * sum((labels' .* log(h) + (1 - labels)' .* (log(1 - h))));
end

% Generating random data points based on a Gaussian Mixture Model
function [x, labels] = randGMM(N, alpha, mu, Sigma)
    d = size(mu, 1);
    cum_alpha = [0, cumsum(alpha)];
    u = rand(1, N);
    x = zeros(d, N);
    labels = zeros(1, N);

    for m = 1:length(alpha)
        ind = find(cum_alpha(m) < u & u <= cum_alpha(m + 1));
        x(:, ind) = randGaussian(length(ind), mu(:, m), Sigma(:, :, m));
        labels(ind) = m - 1;
    end
end

% Generating random data points following a Gaussian distribution
function x = randGaussian(N, mu, Sigma)
    n = length(mu);
    z = randn(n, N);
    A = Sigma^(1/2);
    x = A * z + repmat(mu, 1, N);
end

% Evaluating the GMM for given data points
function gmm = evalGMM(x, alpha, mu, Sigma)
    gmm = zeros(1, size(x, 2));

    for m = 1:length(alpha)
        gmm = gmm + alpha(m) * evalGaussian(x, mu(:, m), Sigma(:, :, m));
    end
end

% Evaluating the Gaussian function for given data points
function g = evalGaussian(x, mu, Sigma)
    [n, N] = size(x);
    invSigma = inv(Sigma);
    C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
    E = -0.5 * sum((x - repmat(mu, 1, N)) .* (invSigma * (x - repmat(mu, 1, N))), 1);
    g = C * exp(E);
end

% Calculating probabilities and error metrics based on decision scores
function prob = CalcProb(discScore, logGamma, labels, N0, N1, phat)
    for ind = 1:length(logGamma)
        prob.decisions = discScore >= logGamma(ind);
        Num_pos(ind) = sum(prob.decisions);
        prob.p10(ind) = sum(prob.decisions == 1 & labels == 0) / N0;
        prob.p11(ind) = sum(prob.decisions == 1 & labels == 1) / N1;
        prob.p01(ind) = sum(prob.decisions == 0 & labels == 1) / N1;
        prob.p00(ind) = sum(prob.decisions == 0 & labels == 0) / N0;
        prob.pFE(ind) = prob.p10(ind) * phat(1) + prob.p01(ind) * phat(2);
    end
end

% Plots contours for estimated GMMs
function plotContours(x, alpha, mu, Sigma)
    figure
    
    if size(x, 1) == 2
        plot(x(1, :), x(2, :), 'b.');
        xlabel('x1');
        ylabel('x2');
        title('Data and Estimated GMM Contours');
        axis equal;
        hold on;
        rangex1 = [min(x(1, :)), max(x(1, :))];
        rangex2 = [min(x(2, :)), max(x(2, :))];
        [x1Grid, x2Grid, zGMM] = contourGMM(alpha, mu, Sigma, rangex1, rangex2);
        contour(x1Grid, x2Grid, zGMM);
        axis equal;
    end
end

% Computing contours for GMMs
function [x1Grid, x2Grid, zGMM] = contourGMM(alpha, mu, Sigma, rangex1, rangex2)
    x1Grid = linspace(floor(rangex1(1)), ceil(rangex1(2)), 101);
    x2Grid = linspace(floor(rangex2(1)), ceil(rangex2(2)), 91);
    [h, v] = meshgrid(x1Grid, x2Grid);
    GMM = evalGMM([h(:)';v(:)'], alpha, mu, Sigma);
    zGMM = reshape(GMM, 91, 101);
end

% Plotting the ROC curve
function plotROC(p10, p11, min_FP, min_TP)
    figure;
    plot(p10, p11, 'DisplayName', 'ROC Curve', 'LineWidth', 2);
    hold on;
    plot(min_FP, min_TP, 'o', 'DisplayName', 'Estimated Min. Error', 'LineWidth', 2);
    xlabel('Prob. False Positive');
    ylabel('Prob. True Positive');
    title('Minimum Expected Risk ROC Curve');
    legend('show');
    grid on;
    box on;
end

% Plots minimum PFE against Gamma
function plotMinPFE(logGamma, pFE, min_pFE_ind)
    figure;
    plot(logGamma, pFE, 'DisplayName', 'Errors', 'LineWidth', 2);
    hold on;
    plot(logGamma(min_pFE_ind), pFE(min_pFE_ind), 'ro', 'DisplayName', 'Minimum Error', 'LineWidth', 2);
    xlabel('Gamma');
    ylabel('Proportion of Errors');
    title('Probability of Error vs. Gamma');
    grid on;
    legend('show');
end

% Plotting data points and classifier decisions
function plotDecisions(x, labels, decisions)
    ind00 = find(decisions == 0 & labels == 0);
    ind10 = find(decisions == 1 & labels == 0);
    ind01 = find(decisions == 0 & labels == 1);
    ind11 = find(decisions == 1 & labels == 1);
    figure;
    plot(x(1, ind00), x(2, ind00), 'og', 'DisplayName', 'Class 0, Correct');
    hold on;
    plot(x(1, ind10), x(2, ind10), 'or', 'DisplayName', 'Class 0, Incorrect');
    hold on;
    plot(x(1, ind01), x(2, ind01), '+r', 'DisplayName', 'Class 1, Correct');
    hold on;
    plot(x(1, ind11), x(2, ind11), '+g', 'DisplayName', 'Class 1, Incorrect');
    hold on;
    axis equal;
    grid on;
    title('Data and respective Classifier Decisions versus True Labels');
    xlabel('x_1');
    ylabel('x_2');
    legend('AutoUpdate', 'off');
    legend('show');
end

% Plotting contours for the Equilibrium Risk Minimization (ERM)
function plotERMContours(x, alpha0, mu0, Sigma0, alpha1, mu1, Sigma1, logGamma_ideal)
    horizontalGrid = linspace(floor(min(x(1, :))), ceil(max(x(1, :))), 101);
    verticalGrid = linspace(floor(min(x(2, :))), ceil(max(x(2, :))), 91);
    [h, v] = meshgrid(horizontalGrid, verticalGrid);
    discriminantScoreGridValues = log(evalGMM([h(:)'; v(:)'], alpha1, mu1, Sigma1)) - log(evalGMM([h(:)'; v(:)'], alpha0, mu0, Sigma0)) - logGamma_ideal;
    minDSGV = min(discriminantScoreGridValues);
    maxDSGV = max(discriminantScoreGridValues);
    discriminantScoreGrid = reshape(discriminantScoreGridValues, 91, 101);
    contour(horizontalGrid, verticalGrid, discriminantScoreGrid, [minDSGV * [0.9, 0.6, 0.3], 0, [0.3, 0.6, 0.9] * maxDSGV]);
    lgd=legend('Correct decisions for data from Class 0', 'Incorrect decisions for data from Class 0', 'Incorrect decisions for data from Class 1', 'Correct decisions for data from Class 1', 'Equilevel contours of the discriminant function');
    set(lgd, 'FontSize', 6); 
end
