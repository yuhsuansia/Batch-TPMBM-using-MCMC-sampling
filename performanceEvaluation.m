function [tgospa_cost, loc_cost, miss_cost, fa_cost, switch_cost] = performanceEvaluation(objectTrajectory,objectTrajectoryEstimate,totalTimeSteps,TGOSPA)

%convert ground truth trajectories to desired format
nx = length(objectTrajectory);
X.tVec = zeros(nx,1);
X.iVec = zeros(nx,1);
X.xState = nan(2,totalTimeSteps,nx);
for i = 1:nx
    X.tVec(i) = objectTrajectory(i).birthTime;
    X.iVec(i) = objectTrajectory(i).endTime-objectTrajectory(i).birthTime+1;
    X.xState(:,objectTrajectory(i).birthTime:objectTrajectory(i).endTime,i) = objectTrajectory(i).stateSequence(1:2,:);
end

%convert estimated trajectories to desired format
ny = length(objectTrajectoryEstimate);
Y.tVec = zeros(ny,1);
Y.iVec = zeros(ny,1);
Y.xState = nan(2,totalTimeSteps,ny);
for i = 1:ny
    Y.tVec(i) = objectTrajectoryEstimate(i).birthTime;
    Y.iVec(i) = objectTrajectoryEstimate(i).endTime-objectTrajectoryEstimate(i).birthTime+1;
    Y.xState(:,objectTrajectoryEstimate(i).birthTime:objectTrajectoryEstimate(i).endTime,i) = objectTrajectoryEstimate(i).stateSequence(1:2,:);
end

[tgospa_cost, loc_cost, miss_cost, fa_cost, switch_cost] = LPTrajMetric_cluster(X, Y, TGOSPA.c, TGOSPA.p, TGOSPA.gamma);
switch_cost = [switch_cost;0];

end

