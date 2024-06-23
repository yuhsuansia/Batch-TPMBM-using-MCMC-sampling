%Tracking scenario with six objects, moving close to each other, staying
%closely-spaced for some time, and moving away from each other

numTrajectory = 6;
objectAppearTime = [1 1 11 11 21 21];
objectDisappearTime = [61 61 71 71 81 81];
objectTrajectory = repmat(struct('birthTime',[],'endTime',[],'stateSequence',[]),[1,numTrajectory]);
midTime = (multiObjectDynamicModel.totalTimeSteps+1)/2;
for i = 1:numTrajectory
    objectTrajectory(i).birthTime = objectAppearTime(i);
    objectTrajectory(i).endTime = objectDisappearTime(i);
    objectTrajectory(i).stateSequence = zeros(multiObjectDynamicModel.stateDimension,multiObjectDynamicModel.totalTimeSteps);
    objectTrajectory(i).stateSequence(:,midTime) = diag([2,2,0,0])*randn(multiObjectDynamicModel.stateDimension,1) + [0;0;2;2];
    for timeStep = midTime+1:objectDisappearTime(i)
        objectTrajectory(i).stateSequence(:,timeStep) = mvnrnd((multiObjectDynamicModel.transitionMatrix*objectTrajectory(i).stateSequence(:,timeStep-1))',multiObjectDynamicModel.motionNoiseCovariance)';
    end
    for timeStep = midTime-1:-1:objectAppearTime(i)
        objectTrajectory(i).stateSequence(:,timeStep) = mvnrnd((multiObjectDynamicModel.transitionMatrix\objectTrajectory(i).stateSequence(:,timeStep+1))',multiObjectDynamicModel.motionNoiseCovariance)';
    end
    objectTrajectory(i).stateSequence = objectTrajectory(i).stateSequence(:,objectAppearTime(i):objectDisappearTime(i));
end

%Poisson birth model with Gaussian mixture intensity
multiObjectDynamicModel.PoissonBirth = repmat(struct('weight',0.01,'mean',[],'covariance',diag([2 2 2 2].^2),'informationVector',initialInformationVector,'informationMatrix',initialInformationMatrix),[1 numTrajectory]);
for i = 1:numTrajectory
    multiObjectDynamicModel.PoissonBirth(i).mean = round(objectTrajectory(i).stateSequence(:,1));
    multiObjectDynamicModel.PoissonBirth(i).informationMatrix(1:multiObjectDynamicModel.stateDimension,1:multiObjectDynamicModel.stateDimension) = multiObjectDynamicModel.PoissonBirth(i).covariance\eye(multiObjectDynamicModel.stateDimension);
    multiObjectDynamicModel.PoissonBirth(i).informationVector(1:multiObjectDynamicModel.stateDimension) = multiObjectDynamicModel.PoissonBirth(i).informationMatrix(1:multiObjectDynamicModel.stateDimension,1:multiObjectDynamicModel.stateDimension)*multiObjectDynamicModel.PoissonBirth(i).mean;
end
multiObjectDynamicModel.PoissonBirthRate = sum([multiObjectDynamicModel.PoissonBirth.weight]);

%multi-Bernoulli birth with Gaussian density
multiObjectDynamicModel.multiBernoulliBirth = repmat(struct('weight',0.01,'mean',[],'covariance',diag([2 2 2 2].^2),'informationVector',initialInformationVector,'informationMatrix',initialInformationMatrix),[1 numTrajectory]);
for i = 1:numTrajectory
    multiObjectDynamicModel.multiBernoulliBirth(i).mean = round(objectTrajectory(i).stateSequence(:,1));
    multiObjectDynamicModel.multiBernoulliBirth(i).informationMatrix(1:multiObjectDynamicModel.stateDimension,1:multiObjectDynamicModel.stateDimension) = multiObjectDynamicModel.multiBernoulliBirth(i).covariance\eye(multiObjectDynamicModel.stateDimension);
    multiObjectDynamicModel.multiBernoulliBirth(i).informationVector(1:multiObjectDynamicModel.stateDimension) = multiObjectDynamicModel.multiBernoulliBirth(i).informationMatrix(1:multiObjectDynamicModel.stateDimension,1:multiObjectDynamicModel.stateDimension)*multiObjectDynamicModel.multiBernoulliBirth(i).mean;
end
