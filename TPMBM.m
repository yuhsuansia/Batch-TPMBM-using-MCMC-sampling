%initialization
trajectoryPPP = [];
trajectoryBernoulli = [];

%memory for storing newly detected trajectories for post-processing
newlyDetectedTrajectory = cell(multiObjectDynamicModel.totalTimeSteps,1);
measurementAssociatedToNew = cell(multiObjectDynamicModel.totalTimeSteps,1);

fprintf('TPMBM filtering time step:');
for timeStep = 1:multiObjectDynamicModel.totalTimeSteps
    fprintf(' %d', timeStep);
    %prediction of undetected trajectories
    numGaussianTrajectoryPPP = length(trajectoryPPP);
    for i = 1:numGaussianTrajectoryPPP
        trajectoryPPP(i).weight = trajectoryPPP(i).weight*multiObjectDynamicModel.survivalProbability;

        trajectoryPPP(i).marginalMean = multiObjectDynamicModel.transitionMatrix*trajectoryPPP(i).marginalMean;
        trajectoryPPP(i).predictMean(:,timeStep) =  trajectoryPPP(i).marginalMean;
        trajectoryPPP(i).marginalCovariance = multiObjectDynamicModel.transitionMatrix*trajectoryPPP(i).marginalCovariance*multiObjectDynamicModel.transitionMatrix' + multiObjectDynamicModel.motionNoiseCovariance;
        trajectoryPPP(i).predictCovariance(:,:,timeStep) =  trajectoryPPP(i).marginalCovariance;

        informationIndex = (1:2*multiObjectDynamicModel.stateDimension)+multiObjectDynamicModel.stateDimension*(trajectoryPPP(i).endTime-trajectoryPPP(i).birthTime);
        trajectoryPPP(i).informationMatrix(informationIndex,informationIndex) = full(trajectoryPPP(i).informationMatrix(informationIndex,informationIndex)) + multiObjectDynamicModel.predictionInformationMatrix;

        trajectoryPPP(i).endTime = trajectoryPPP(i).endTime + 1;
    end

    %add newborn trajectories
    for i = 1:length(multiObjectDynamicModel.PoissonBirth)
        trajectoryPPP(numGaussianTrajectoryPPP+i).weight = multiObjectDynamicModel.PoissonBirth(i).weight;
        trajectoryPPP(numGaussianTrajectoryPPP+i).marginalMean = multiObjectDynamicModel.PoissonBirth(i).mean;
        trajectoryPPP(numGaussianTrajectoryPPP+i).predictMean = zeros(multiObjectDynamicModel.stateDimension,multiObjectDynamicModel.totalTimeSteps);
        trajectoryPPP(numGaussianTrajectoryPPP+i).predictMean(:,timeStep) = multiObjectDynamicModel.PoissonBirth(i).mean;
        trajectoryPPP(numGaussianTrajectoryPPP+i).marginalCovariance = multiObjectDynamicModel.PoissonBirth(i).covariance;
        trajectoryPPP(numGaussianTrajectoryPPP+i).predictCovariance = zeros(multiObjectDynamicModel.stateDimension,multiObjectDynamicModel.stateDimension,multiObjectDynamicModel.totalTimeSteps);
        trajectoryPPP(numGaussianTrajectoryPPP+i).predictCovariance(:,:,timeStep) = multiObjectDynamicModel.PoissonBirth(i).covariance;
        trajectoryPPP(numGaussianTrajectoryPPP+i).informationVector = multiObjectDynamicModel.PoissonBirth(i).informationVector;
        trajectoryPPP(numGaussianTrajectoryPPP+i).informationMatrix = multiObjectDynamicModel.PoissonBirth(i).informationMatrix;
        trajectoryPPP(numGaussianTrajectoryPPP+i).birthTime = timeStep;
        trajectoryPPP(numGaussianTrajectoryPPP+i).endTime = timeStep;
    end

    %prediction of detected trajectories
    numBernoulli = length(trajectoryBernoulli);
    for i = 1:numBernoulli
        numLocalHypothesis = length(trajectoryBernoulli{i});
        for h = 1:numLocalHypothesis
            localHypothesis = trajectoryBernoulli{i}(h);
            if localHypothesis.isAlive
                %each local hypothesis has a mixture representation
                for l = 1:length(localHypothesis.trajectoryMixture)
                    if localHypothesis.endTimeProbability(l,timeStep-1)*localHypothesis.existenceProbability >= minimumEndTimeProbability
                        localHypothesis.trajectoryMixture(l).marginalMean = multiObjectDynamicModel.transitionMatrix*localHypothesis.trajectoryMixture(l).marginalMean;
                        localHypothesis.trajectoryMixture(l).marginalCovariance = multiObjectDynamicModel.transitionMatrix*localHypothesis.trajectoryMixture(l).marginalCovariance*multiObjectDynamicModel.transitionMatrix' + multiObjectDynamicModel.motionNoiseCovariance;

                        informationIndex = (1:2*multiObjectDynamicModel.stateDimension)+multiObjectDynamicModel.stateDimension*(localHypothesis.endTime(l)-localHypothesis.birthTime(l));
                        localHypothesis.trajectoryMixture(l).informationMatrix(informationIndex,informationIndex) = full(localHypothesis.trajectoryMixture(l).informationMatrix(informationIndex,informationIndex)) + multiObjectDynamicModel.predictionInformationMatrix;

                        localHypothesis.endTime(l) = localHypothesis.endTime(l) + 1;
                        localHypothesis.endTimeProbability(l,timeStep) = localHypothesis.endTimeProbability(l,timeStep-1)*multiObjectDynamicModel.survivalProbability;
                        localHypothesis.endTimeProbability(l,timeStep-1) = localHypothesis.endTimeProbability(l,timeStep-1)*(1-multiObjectDynamicModel.survivalProbability);
                    end
                end
            end
            trajectoryBernoulli{i}(h) = localHypothesis;
        end
    end

    %update of detected trajectories
    z = measurements{timeStep};
    numMeasurements = size(z,2);

    updatedTrajectoryBernoulli = cell(1,numBernoulli);
    updatedLogLikelihoodTable = cell(1,numBernoulli);
    for i = 1:numBernoulli
        numLocalHypothesis = length(trajectoryBernoulli{i});
        updatedTrajectoryBernoulli{i} = repmat(struct('trajectoryMixture',[],'birthTime',[],'birthTimeProbability',[],'endTime',[],'endTimeProbability',0,'likelihood',[],'existenceProbability',[],'associationHistory',[],'firstDetectedTimeStep',[],'isAlive',[]),[1,numLocalHypothesis*(numMeasurements+1)]);
        updatedLogLikelihoodTable{i} = -inf(numLocalHypothesis,1+numMeasurements);

        %go through each predicted local hypothesis for each Bernoulli
        for h = 1:numLocalHypothesis
            localHypothesis = trajectoryBernoulli{i}(h);

            %misdetection local hypothesis
            misDetectionHypothesis = localHypothesis;

            aliveProbability = 0;
            numTrajectoryMixture = length(misDetectionHypothesis.trajectoryMixture);
            if misDetectionHypothesis.isAlive
                for l = 1:numTrajectoryMixture
                    conditionalAliveProbability = misDetectionHypothesis.endTimeProbability(l,timeStep);
                    if conditionalAliveProbability*misDetectionHypothesis.existenceProbability >= minimumEndTimeProbability
                        misDetectionHypothesis.endTimeProbability(l,timeStep) = misDetectionHypothesis.endTimeProbability(l,timeStep)*(1-multiObjectMeasurementModel.detectionProbability);
                        misDetectionHypothesis.endTimeProbability(l,:) = misDetectionHypothesis.endTimeProbability(l,:)/(1-multiObjectMeasurementModel.detectionProbability*conditionalAliveProbability);

                        misDetectionHypothesis.trajectoryMixture(l).filterMean(:,timeStep) = misDetectionHypothesis.trajectoryMixture(l).marginalMean;
                        misDetectionHypothesis.trajectoryMixture(l).filterCovariance(:,:,timeStep) = misDetectionHypothesis.trajectoryMixture(l).marginalCovariance;

                        aliveProbability = aliveProbability + multiObjectMeasurementModel.detectionProbability*conditionalAliveProbability*misDetectionHypothesis.birthTimeProbability(l);
                    end
                end
            end

            updatedLogLikelihoodTable{i}(h,1) = log(1-misDetectionHypothesis.existenceProbability*aliveProbability);
            misDetectionHypothesis.likelihood = misDetectionHypothesis.likelihood + updatedLogLikelihoodTable{i}(h,1);
            misDetectionHypothesis.existenceProbability = misDetectionHypothesis.existenceProbability*(1-aliveProbability)/(1-misDetectionHypothesis.existenceProbability*aliveProbability);

            updatedTrajectoryBernoulli{i}((h-1)*(1+numMeasurements)+1) = misDetectionHypothesis;

            %precompute
            innovationCovariance = zeros(multiObjectMeasurementModel.measurementDimension,multiObjectMeasurementModel.measurementDimension,numTrajectoryMixture);
            innovationCovarianceInverse = zeros(multiObjectMeasurementModel.measurementDimension,multiObjectMeasurementModel.measurementDimension,numTrajectoryMixture);
            innovationCovarianceDeterminant = zeros(numTrajectoryMixture,1);
            predictedMeasurement = zeros(multiObjectMeasurementModel.measurementDimension,numTrajectoryMixture);
            KalmanGain = zeros(multiObjectDynamicModel.stateDimension,multiObjectMeasurementModel.measurementDimension,numTrajectoryMixture);
            marginalCovariance = zeros(multiObjectDynamicModel.stateDimension,multiObjectDynamicModel.stateDimension,numTrajectoryMixture);

            if localHypothesis.isAlive
                for l = 1:numTrajectoryMixture
                    trajectory = localHypothesis.trajectoryMixture(l);
                    if localHypothesis.endTimeProbability(l,timeStep)*localHypothesis.existenceProbability >= minimumEndTimeProbability
                        innovationCovariance(:,:,l) = multiObjectMeasurementModel.observationMatrix*trajectory.marginalCovariance*multiObjectMeasurementModel.observationMatrix' + multiObjectMeasurementModel.measurementNoiseCovariance;
                        innovationCovarianceInverse(:,:,l) = innovationCovariance(:,:,l)\eye(multiObjectMeasurementModel.measurementDimension);
                        innovationCovarianceDeterminant(l) = det(innovationCovariance(:,:,l));
                        predictedMeasurement(:,l) = multiObjectMeasurementModel.observationMatrix*trajectory.marginalMean;
                        KalmanGain(:,:,l) = trajectory.marginalCovariance*multiObjectMeasurementModel.observationMatrix'*innovationCovarianceInverse(:,:,l);
                        marginalCovariance(:,:,l) = (eye(multiObjectDynamicModel.stateDimension) - KalmanGain(:,:,l)*multiObjectMeasurementModel.observationMatrix)*trajectory.marginalCovariance;
                        marginalCovariance(:,:,l) = (marginalCovariance(:,:,l)+marginalCovariance(:,:,l)')/2;
                    end
                end
            end

            %measurement update local hypothesis
            for j = 1:numMeasurements
                measurementUpdateHypothesis = localHypothesis;

                isInGate = false(numTrajectoryMixture,1);
                associationLikelihood = zeros(numTrajectoryMixture,1);
                if measurementUpdateHypothesis.isAlive
                    for l = 1:numTrajectoryMixture
                        trajectory = measurementUpdateHypothesis.trajectoryMixture(l);
                        if measurementUpdateHypothesis.endTimeProbability(l,timeStep)*measurementUpdateHypothesis.existenceProbability >= minimumEndTimeProbability
                            innovation = z(:,j)-predictedMeasurement(:,l);
                            mahalanobisDistance = innovation'*innovationCovarianceInverse(:,:,l)*innovation;

                            if mahalanobisDistance < gateSize
                                isInGate(l) = true;

                                trajectory.marginalMean = trajectory.marginalMean + KalmanGain(:,:,l)*innovation;
                                trajectory.marginalCovariance = marginalCovariance(:,:,l);

                                trajectory.filterMean(:,timeStep) = trajectory.marginalMean;
                                trajectory.filterCovariance(:,:,timeStep) = trajectory.marginalCovariance;

                                informationIndex = (1:2)+(measurementUpdateHypothesis.endTime(l)-measurementUpdateHypothesis.birthTime(l))*multiObjectDynamicModel.stateDimension;
                                trajectory.informationMatrix(informationIndex,informationIndex) = full(trajectory.informationMatrix(informationIndex,informationIndex)) + multiObjectMeasurementModel.inverseMeasurementNoiseCovariance;
                                trajectory.informationVector(informationIndex) = trajectory.informationVector(informationIndex) + multiObjectMeasurementModel.inverseMeasurementNoiseCovariance*z(:,j);

                                measurementLikelihood = exp(-mahalanobisDistance/2)/sqrt((2*pi)^multiObjectMeasurementModel.measurementDimension*innovationCovarianceDeterminant(l));
                                associationLikelihood(l) = measurementUpdateHypothesis.birthTimeProbability(l)*measurementUpdateHypothesis.endTimeProbability(l,timeStep)*multiObjectMeasurementModel.detectionProbability*measurementLikelihood;
                                measurementUpdateHypothesis.endTimeProbability(l,:) = zeros(1,multiObjectDynamicModel.totalTimeSteps);
                                measurementUpdateHypothesis.endTimeProbability(l,timeStep) = 1;
                                measurementUpdateHypothesis.trajectoryMixture(l) = trajectory;
                            end
                        end
                    end
                end

                if any(isInGate)
                    measurementUpdateHypothesis.trajectoryMixture = measurementUpdateHypothesis.trajectoryMixture(isInGate);
                    measurementUpdateHypothesis.birthTime = measurementUpdateHypothesis.birthTime(isInGate);
                    measurementUpdateHypothesis.birthTimeProbability = associationLikelihood(isInGate)/sum(associationLikelihood);
                    measurementUpdateHypothesis.endTime = measurementUpdateHypothesis.endTime(isInGate);
                    measurementUpdateHypothesis.endTimeProbability = measurementUpdateHypothesis.endTimeProbability(isInGate,:);

                    %remove mixture components with small birth time probability
                    keepIndex = measurementUpdateHypothesis.birthTimeProbability > minimumBirthTimeProbability;
                    measurementUpdateHypothesis.birthTime = measurementUpdateHypothesis.birthTime(keepIndex);
                    measurementUpdateHypothesis.birthTimeProbability = measurementUpdateHypothesis.birthTimeProbability(keepIndex);
                    measurementUpdateHypothesis.birthTimeProbability = measurementUpdateHypothesis.birthTimeProbability/sum(measurementUpdateHypothesis.birthTimeProbability);
                    measurementUpdateHypothesis.endTime = measurementUpdateHypothesis.endTime(keepIndex);
                    measurementUpdateHypothesis.endTimeProbability = measurementUpdateHypothesis.endTimeProbability(keepIndex,:);
                    measurementUpdateHypothesis.trajectoryMixture = measurementUpdateHypothesis.trajectoryMixture(keepIndex);

                    updatedLogLikelihoodTable{i}(h,1+j) = log(measurementUpdateHypothesis.existenceProbability*sum(associationLikelihood));
                    measurementUpdateHypothesis.likelihood = measurementUpdateHypothesis.likelihood + updatedLogLikelihoodTable{i}(h,1+j);
                    measurementUpdateHypothesis.existenceProbability = 1;
                    measurementUpdateHypothesis.associationHistory(timeStep) = j;

                    updatedTrajectoryBernoulli{i}((h-1)*(1+numMeasurements)+1+j) = measurementUpdateHypothesis;
                end
            end
        end
    end

    %precompute
    numGaussianTrajectoryPPP = length(trajectoryPPP);
    innovationCovariance = zeros(multiObjectMeasurementModel.measurementDimension,multiObjectMeasurementModel.measurementDimension,numGaussianTrajectoryPPP);
    innovationCovarianceInverse = zeros(multiObjectMeasurementModel.measurementDimension,multiObjectMeasurementModel.measurementDimension,numGaussianTrajectoryPPP);
    innovationCovarianceDeterminant = zeros(numGaussianTrajectoryPPP,1);
    predictedMeasurement = zeros(multiObjectMeasurementModel.measurementDimension,numGaussianTrajectoryPPP);
    KalmanGain = zeros(multiObjectDynamicModel.stateDimension,multiObjectMeasurementModel.measurementDimension,numGaussianTrajectoryPPP);
    marginalCovariance = zeros(multiObjectDynamicModel.stateDimension,multiObjectDynamicModel.stateDimension,numGaussianTrajectoryPPP);

    for l = 1:numGaussianTrajectoryPPP
        trajectory = trajectoryPPP(l);
        innovationCovariance(:,:,l) = multiObjectMeasurementModel.observationMatrix*trajectory.marginalCovariance*multiObjectMeasurementModel.observationMatrix' + multiObjectMeasurementModel.measurementNoiseCovariance;
        innovationCovarianceInverse(:,:,l) = innovationCovariance(:,:,l)\eye(multiObjectMeasurementModel.measurementDimension);
        innovationCovarianceDeterminant(l) = det(innovationCovariance(:,:,l));
        predictedMeasurement(:,l) = multiObjectMeasurementModel.observationMatrix*trajectory.marginalMean;
        KalmanGain(:,:,l) = trajectory.marginalCovariance*multiObjectMeasurementModel.observationMatrix'*innovationCovarianceInverse(:,:,l);
        marginalCovariance(:,:,l) = (eye(multiObjectDynamicModel.stateDimension) - KalmanGain(:,:,l)*multiObjectMeasurementModel.observationMatrix)*trajectory.marginalCovariance;
        marginalCovariance(:,:,l) = (marginalCovariance(:,:,l)+marginalCovariance(:,:,l)')/2;
    end

    %measurement update of undetected trajectories
    updatedNewTrajectoryBernoulli = cell(1,numMeasurements);
    costMatrixNew = inf(numMeasurements,numMeasurements);
    initialTrajectoryMixture = repmat(struct('marginalMean',[],'marginalCovariance',[],'informationVector',[],'informationMatrix',[],'filterMean',[],'filterCovariance',[]),[1 numGaussianTrajectoryPPP]);
    for j = 1:numMeasurements
        isInGate = false(numGaussianTrajectoryPPP,1);
        associationLikelihood = zeros(numGaussianTrajectoryPPP,1);

        trajectoryMixture = initialTrajectoryMixture;
        newBirthTime = zeros(1,numGaussianTrajectoryPPP);
        newEndTime = zeros(1,numGaussianTrajectoryPPP);
        newEndTimeProbability = zeros(numGaussianTrajectoryPPP,multiObjectDynamicModel.totalTimeSteps);
        for i = 1:numGaussianTrajectoryPPP
            innovation = z(:,j)-predictedMeasurement(:,i);
            mahalanobisDistance = innovation'*innovationCovarianceInverse(:,:,i)*innovation;
            if mahalanobisDistance < gateSize
                isInGate(i) = true;
                trajectory = trajectoryPPP(i);

                newBirthTime(i) = trajectory.birthTime;
                newEndTime(i) = trajectory.endTime;
                newEndTimeProbability(i,timeStep) = 1;

                trajectory.marginalCovariance = marginalCovariance(:,:,i);
                innovation = z(:,j)-predictedMeasurement(:,i);
                trajectory.marginalMean = trajectory.marginalMean + KalmanGain(:,:,i)*innovation;
                trajectoryMixture(i).marginalMean = trajectory.marginalMean;
                trajectoryMixture(i).marginalCovariance = trajectory.marginalCovariance;

                trajectoryMixture(i).filterMean = trajectory.predictMean;
                trajectoryMixture(i).filterMean(:,timeStep) = trajectory.marginalMean;
                trajectoryMixture(i).filterCovariance = trajectory.predictCovariance;
                trajectoryMixture(i).filterCovariance(:,:,timeStep) = trajectory.marginalCovariance;

                informationIndex = (1:2)+(trajectory.endTime-trajectory.birthTime)*multiObjectDynamicModel.stateDimension;
                trajectory.informationMatrix(informationIndex,informationIndex) = full(trajectory.informationMatrix(informationIndex,informationIndex)) + multiObjectMeasurementModel.inverseMeasurementNoiseCovariance;
                trajectory.informationVector(informationIndex) = trajectory.informationVector(informationIndex) + multiObjectMeasurementModel.inverseMeasurementNoiseCovariance*z(:,j);
                trajectoryMixture(i).informationVector = trajectory.informationVector;
                trajectoryMixture(i).informationMatrix = trajectory.informationMatrix;

                measurementLikelihood = exp(-mahalanobisDistance/2)/sqrt((2*pi)^multiObjectMeasurementModel.measurementDimension*innovationCovarianceDeterminant(l));
                associationLikelihood(i) = trajectory.weight*multiObjectMeasurementModel.detectionProbability*measurementLikelihood;
            end
        end

        if any(isInGate)
            newHypothesis.trajectoryMixture = trajectoryMixture(isInGate);
            newHypothesis.birthTime = newBirthTime(isInGate);
            newHypothesis.birthTimeProbability = associationLikelihood(isInGate)/sum(associationLikelihood);
            newHypothesis.endTime = newEndTime(isInGate);
            newHypothesis.endTimeProbability = newEndTimeProbability(isInGate,:);

            %remove mixture components with small birth time probability
            keepIndex = newHypothesis.birthTimeProbability > minimumBirthTimeProbability;
            newHypothesis.birthTime = newHypothesis.birthTime(keepIndex);
            newHypothesis.birthTimeProbability = newHypothesis.birthTimeProbability(keepIndex);
            newHypothesis.birthTimeProbability = newHypothesis.birthTimeProbability/sum(newHypothesis.birthTimeProbability);
            newHypothesis.endTime = newHypothesis.endTime(keepIndex);
            newHypothesis.endTimeProbability = newHypothesis.endTimeProbability(keepIndex,:);

            newHypothesis.trajectoryMixture = newHypothesis.trajectoryMixture(keepIndex);

            newHypothesis.likelihood = log(sum(associationLikelihood) + multiObjectMeasurementModel.PoissonClutterIntensity);
            newHypothesis.existenceProbability = sum(associationLikelihood)/exp(newHypothesis.likelihood);

            newHypothesis.associationHistory = zeros(1,multiObjectDynamicModel.totalTimeSteps);
            newHypothesis.associationHistory(timeStep) = j;
            newHypothesis.firstDetectedTimeStep = timeStep;
            newHypothesis.isAlive = true;

            updatedNewTrajectoryBernoulli{j}(1) = newHypothesis;
            costMatrixNew(j,j) = -newHypothesis.likelihood;
        else
            updatedNewTrajectoryBernoulli{j}(1) = struct('trajectoryMixture',[],'birthTime',[],'birthTimeProbability',[],'endTime',[],'endTimeProbability',[],'likelihood',[],'existenceProbability',0,'associationHistory',[],'firstDetectedTimeStep',[],'isAlive',[]);
            costMatrixNew(j,j) = -log(multiObjectMeasurementModel.PoissonClutterIntensity);
        end
    end

    newlyDetectedTrajectory{timeStep} = updatedNewTrajectoryBernoulli;
    measurementAssociatedToNew{timeStep} = cellfun(@(x) x.existenceProbability > minimumExistenceProbability, updatedNewTrajectoryBernoulli);

    %construct global hypothesis
    numUpdatedBernoulli = numBernoulli + numMeasurements;
    if numBernoulli == 0
        updatedGlobalHypothesisLogWeight = 0;
        updatedGlobalHypothesisLookUpTable = ones(1,numMeasurements);
    else
        updatedGlobalHypothesisLogWeight = [];
        updatedGlobalHypothesisLookUpTable = [];
        numGlobalHypothesis = length(globalHypothesisLogWeight);

        %go through each predicted global hypothesis
        for a = 1:numGlobalHypothesis
            %construct cost matrix
            costMatrixOld = inf(numMeasurements,numBernoulli);
            costMisdetectionSum = 0;
            for i = 1:numBernoulli
                if globalHypothesisLookUpTable(a,i)~=0
                    costMisdetection = updatedLogLikelihoodTable{i}(globalHypothesisLookUpTable(a,i),1);
                    costMisdetectionSum = costMisdetectionSum + costMisdetection;
                    costMatrixOld(:,i) = -updatedLogLikelihoodTable{i}(globalHypothesisLookUpTable(a,i),2:end) + costMisdetection;
                end
            end
            costMatrix = [costMatrixOld costMatrixNew];

            %find the k-best assignments
            [col4rowBest,~,gainBest] = kBest2DAssignMATLAB(costMatrix,ceil(maxiNumGlobalHypothesis*exp(globalHypothesisLogWeight(a))));
            updatedGlobalHypothesisLogWeight = [updatedGlobalHypothesisLogWeight -gainBest'+costMisdetectionSum+globalHypothesisLogWeight(a)];

            %construct new look up table per predicted global hypothesis
            numAssignments = length(gainBest);
            lookUpTable = zeros(numAssignments,numUpdatedBernoulli);
            for h = 1:numAssignments
                IsmisDetection = true(numBernoulli,1);
                for j = 1:numMeasurements
                    if col4rowBest(j,h) <= numBernoulli
                        IsmisDetection(col4rowBest(j,h)) = false;
                        if globalHypothesisLookUpTable(a,col4rowBest(j,h))
                            lookUpTable(h,col4rowBest(j,h)) = (globalHypothesisLookUpTable(a,col4rowBest(j,h))-1)*(1+numMeasurements)+1+j;
                        end
                    else
                        lookUpTable(h,col4rowBest(j,h)) = 1;
                    end
                end
                for i = 1:numBernoulli
                    if globalHypothesisLookUpTable(a,i) && IsmisDetection(i)
                        lookUpTable(h,i) = (globalHypothesisLookUpTable(a,i)-1)*(1+numMeasurements)+1;
                    end
                end
            end
            updatedGlobalHypothesisLookUpTable = [updatedGlobalHypothesisLookUpTable;lookUpTable];
        end
        updatedGlobalHypothesisLogWeight = normalizeLogWeights(updatedGlobalHypothesisLogWeight);
    end

    %remove global hypothesis with small weights
    keepIndex = updatedGlobalHypothesisLogWeight > log(minimumGlobalHypothesisWeight);
    updatedGlobalHypothesisLogWeight = normalizeLogWeights(updatedGlobalHypothesisLogWeight(keepIndex));
    updatedGlobalHypothesisLookUpTable = updatedGlobalHypothesisLookUpTable(keepIndex,:);

    %cap the number of global hypotheses
    if length(updatedGlobalHypothesisLogWeight) > maxiNumGlobalHypothesis
        [updatedGlobalHypothesisLogWeight,sortedOrder] = sort(updatedGlobalHypothesisLogWeight,'descend');
        updatedGlobalHypothesisLogWeight = normalizeLogWeights(updatedGlobalHypothesisLogWeight(1:maxiNumGlobalHypothesis));
        updatedGlobalHypothesisLookUpTable = updatedGlobalHypothesisLookUpTable(sortedOrder(1:maxiNumGlobalHypothesis),:);
    end

    trajectoryBernoulli = [updatedTrajectoryBernoulli updatedNewTrajectoryBernoulli];

    %remove empty local hypothesis trees
    keepIndex = sum(updatedGlobalHypothesisLookUpTable,1) >= 1;
    updatedGlobalHypothesisLookUpTable = updatedGlobalHypothesisLookUpTable(:,keepIndex);
    trajectoryBernoulli = trajectoryBernoulli(keepIndex);

    %remove local hypotheses not appearing in truncated global hypotheses
    numUpdatedBernoulli = size(updatedGlobalHypothesisLookUpTable,2);
    for i = 1:numUpdatedBernoulli
        globalHypothesisIndex = updatedGlobalHypothesisLookUpTable(:,i);
        trajectoryBernoulli{i} = trajectoryBernoulli{i}(unique(globalHypothesisIndex(globalHypothesisIndex~=0), 'stable'));
    end

    %reindex global hypothesis look up table
    numBernoulli = length(trajectoryBernoulli);
    for i = 1:numBernoulli
        globalHypothesisIndex = updatedGlobalHypothesisLookUpTable(:,i) > 0;
        [~,~,updatedGlobalHypothesisLookUpTable(globalHypothesisIndex,i)] = unique(updatedGlobalHypothesisLookUpTable(globalHypothesisIndex,i),'stable');
    end

    %remove local hypothesis with small existence probability
    for i = 1:numUpdatedBernoulli
        removeIndex = [trajectoryBernoulli{i}.existenceProbability] <= minimumExistenceProbability;
        trajectoryBernoulli{i} = trajectoryBernoulli{i}(~removeIndex);
        removeIndex = find(removeIndex);
        for j = 1:length(removeIndex)
            globalHypothesisIndex = updatedGlobalHypothesisLookUpTable(:,i);
            globalHypothesisIndex(globalHypothesisIndex==removeIndex(j)) = 0;
            updatedGlobalHypothesisLookUpTable(:,i) = globalHypothesisIndex;
        end
    end

    %remove empty local hypothesis trees
    keepIndex = sum(updatedGlobalHypothesisLookUpTable,1) >= 1;
    updatedGlobalHypothesisLookUpTable = updatedGlobalHypothesisLookUpTable(:,keepIndex);
    trajectoryBernoulli = trajectoryBernoulli(keepIndex);

    %reindex global hypothesis look up table
    numBernoulli = length(trajectoryBernoulli);
    for i = 1:numBernoulli
        globalHypothesisIndex = updatedGlobalHypothesisLookUpTable(:,i) > 0;
        [~,~,updatedGlobalHypothesisLookUpTable(globalHypothesisIndex,i)] = unique(updatedGlobalHypothesisLookUpTable(globalHypothesisIndex,i),'stable');
    end

    %remove tracks that have only been detected once and with low probability of being alive
    keepIndex = false(numBernoulli,1);
    for i = 1:numBernoulli
        numLocalHypothesis = length(trajectoryBernoulli{i});
        for h = 1:numLocalHypothesis
            if trajectoryBernoulli{i}(h).existenceProbability < 1
                if any(trajectoryBernoulli{i}(h).endTimeProbability(:,timeStep) > minimumEndTimeProbability)
                    keepIndex(i) = true;
                    break;
                end
            else
                keepIndex(i) = true;
                break;
            end
        end
    end
    trajectoryBernoulli = trajectoryBernoulli(keepIndex);
    updatedGlobalHypothesisLookUpTable = updatedGlobalHypothesisLookUpTable(:,keepIndex);
    numBernoulli = length(trajectoryBernoulli);

    %merge duplicate rows of look up table
    if length(updatedGlobalHypothesisLogWeight) > 1
        [globalHypothesisLookUpTable,~,IC] = unique(updatedGlobalHypothesisLookUpTable,'rows','stable');
        numGlobalHypothesis = size(globalHypothesisLookUpTable,2);
        if numGlobalHypothesis ~= size(updatedGlobalHypothesisLookUpTable,2)
            globalHypothesisLogWeight = zeros(1,numGlobalHypothesis);
            for i = 1:numGlobalHypothesis
                [~,globalHypothesisLogWeight(i)] = normalizeLogWeights(updatedGlobalHypothesisLogWeight(IC==i));
            end
        else
            globalHypothesisLogWeight = updatedGlobalHypothesisLogWeight;
            globalHypothesisLookUpTable = updatedGlobalHypothesisLookUpTable;
        end
    else
        globalHypothesisLogWeight = updatedGlobalHypothesisLogWeight;
        globalHypothesisLookUpTable = updatedGlobalHypothesisLookUpTable;
    end

    %find local hypotheses that are not alive
    for i = 1:numBernoulli
        numLocalHypothesis = length(trajectoryBernoulli{i});
        for h = 1:numLocalHypothesis
            [~,index] = max(trajectoryBernoulli{i}(h).birthTimeProbability);
            if trajectoryBernoulli{i}(h).endTimeProbability(index,timeStep)*trajectoryBernoulli{i}(h).existenceProbability < minimumEndTimeProbability
                trajectoryBernoulli{i}(h).isAlive = false;
            end
        end
    end

    %misdetection of undetected trajectories
    for i = 1:numGaussianTrajectoryPPP
        trajectoryPPP(i).weight = trajectoryPPP(i).weight*(1-multiObjectMeasurementModel.detectionProbability);
    end

    %remove mixture components in Poisson intensity with small weights
    keepIndex = [trajectoryPPP.weight] > minimumPoissonGaussianWeight;
    trajectoryPPP = trajectoryPPP(keepIndex);

    %extract trajectory estimates from the global hypothesis with the highest weight
    [~,index] = max(globalHypothesisLogWeight);
    globalHypothesisMAP = globalHypothesisLookUpTable(index,:);
    objectTrajectoryEstimate = [];
    for i = 1:numBernoulli
        if globalHypothesisMAP(i) > 0
            localHypothesis = trajectoryBernoulli{i}(globalHypothesisMAP(i));
            if localHypothesis.existenceProbability >= estimateExistenceProbabilityThreshold
                [~,birthTimeIndex] = max(localHypothesis.birthTimeProbability);
                trajectoryComponent = localHypothesis.trajectoryMixture(birthTimeIndex);

                maxTrajectoryLength = localHypothesis.endTime(birthTimeIndex)-localHypothesis.birthTime(birthTimeIndex)+1;
                informationIndex = 1:maxTrajectoryLength*multiObjectDynamicModel.stateDimension;
                stateSequence = reshape(trajectoryComponent.informationMatrix(informationIndex,informationIndex)\trajectoryComponent.informationVector(informationIndex),[multiObjectDynamicModel.stateDimension,maxTrajectoryLength]);

                objectTrajectoryEstimate(end+1).birthTime = localHypothesis.birthTime(birthTimeIndex);
                [~,objectTrajectoryEstimate(end).endTime] = max(localHypothesis.endTimeProbability(birthTimeIndex,:));
                objectTrajectoryEstimate(end).stateSequence = stateSequence(:,1:objectTrajectoryEstimate(end).endTime-localHypothesis.birthTime(birthTimeIndex)+1);
                objectTrajectoryEstimate(end).measurementSequence = localHypothesis.associationHistory(objectTrajectoryEstimate(end).birthTime:objectTrajectoryEstimate(end).endTime);
            end
        end
    end
end
fprintf('\nFinished\n');