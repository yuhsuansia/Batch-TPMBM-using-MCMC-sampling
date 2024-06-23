clear;clc

multiObjectModel;
trackingScenarioChallenge;

%parameters
gateSize = chi2inv(0.999,multiObjectMeasurementModel.measurementDimension);
minimumBirthTimeProbability = 1e-1;
minimumEndTimeProbability = 1e-4;
minimumExistenceProbability = 1e-4;
minimumPoissonGaussianWeight = 1e-4;
minimumGlobalHypothesisWeight = 1e-4;
maxiNumGlobalHypothesis = 1e3;
estimateExistenceProbabilityThreshold = 1;
numMinimumAssignment = 1e2;
numFullMetropolisHastingsIteration = 2e6;

TGOSPA.c = 10;
TGOSPA.p = 1;
TGOSPA.gamma = 2;

%generate measurements
generateMeasurments;

%TPMBM filtering
TPMBM;

postProcessing;

sampleMultiBernoulli = trajectoryMultiBernoulli;
sampleMetropolisHastingsLikelihoodDelta = zeros(numFullMetropolisHastingsIteration,1);
mostLikelyMultiBernoulli = sampleMultiBernoulli;
highestLikelihood = 0;

fprintf('MCMC iteration:');
for iter = 1:numFullMetropolisHastingsIteration
    if rem(iter,100) == 0
        fprintf(' %d', iter);
    end
    %sample an action from {update, split, merge, switch}
    numBernoulli = length(sampleMultiBernoulli);
    samplingPool = find(arrayfun(@(x) x.existenceProbability==1, sampleMultiBernoulli));
    numSampleBernoulli = length(samplingPool);
    if numSampleBernoulli > 1
        action = datasample([1 2 3 4 4 4],1);
    else
        action = datasample([1 3],1);
    end
    switch action
        %update action
        case 1
            %sample a Bernoulli
            i = samplingPool(randi(numSampleBernoulli,1));
            firstDetectedTimeStep = sampleMultiBernoulli(i).firstDetectedTimeStep;
            associationHistory = sampleMultiBernoulli(i).associationHistory;

            indexRemove = [];
            newMultiBernoulli = [];
            associationMatrix = reshape([sampleMultiBernoulli.associationHistory],[multiObjectDynamicModel.totalTimeSteps length(sampleMultiBernoulli)]);
            indexNew = sum(associationMatrix>0,1) == 1;

            %go through each possible time step
            timeStep = randi([firstDetectedTimeStep+1,max(sampleMultiBernoulli(i).endTime)],1);

            %find clutter measurements
            measurementAssignment = associationMatrix(timeStep,:);
            unusedMeasurement = true(1,size(measurements{timeStep},2));
            unusedMeasurement(measurementAssignment(measurementAssignment>0)) = false;
            unusedMeasurement = find(unusedMeasurement);

            %find measurements used for initiating new Bernoullis
            indexNewlyDetectedBernoulli = find(indexNew & associationMatrix(timeStep,:) > 0);
            newlyDetectedBernoulli = sampleMultiBernoulli(indexNewlyDetectedBernoulli);
            newDetection = associationMatrix(timeStep,indexNewlyDetectedBernoulli);
            unusedMeasurement = [unusedMeasurement newDetection];

            pastBernoulli = sampleMultiBernoulli(i).past{timeStep-1};
            pastBernoulli.past = sampleMultiBernoulli(i).past;
            measurementAssociation = associationHistory(timeStep:end);
            currentMeasurementAssociation = measurementAssociation(1);
            %consider misdetection
            if currentMeasurementAssociation > 0
                unusedMeasurement = [unusedMeasurement 0];
            end

            %pre-compute parameters for ellipsoidal gating
            numTrajectoryMixture = length(pastBernoulli.trajectoryMixture);
            predictedMean = zeros(multiObjectMeasurementModel.measurementDimension,numTrajectoryMixture);
            inverseInnovationCovariance = zeros(multiObjectMeasurementModel.measurementDimension,multiObjectMeasurementModel.measurementDimension,numTrajectoryMixture);
            for l = 1:numTrajectoryMixture
                predictedMean(:,l) = multiObjectMeasurementModel.observationMatrix*(multiObjectDynamicModel.transitionMatrix*pastBernoulli.trajectoryMixture(l).marginalMean);
                innovationCovariance = multiObjectMeasurementModel.observationMatrix*(multiObjectDynamicModel.transitionMatrix*pastBernoulli.trajectoryMixture(l).marginalCovariance*multiObjectDynamicModel.transitionMatrix' + multiObjectDynamicModel.motionNoiseCovariance)*multiObjectMeasurementModel.observationMatrix' + multiObjectMeasurementModel.measurementNoiseCovariance;
                inverseInnovationCovariance(:,:,l) = inv(innovationCovariance);
            end
            currentMeasurements = measurements{timeStep};

            numPossibleSampledBernoulli = length(unusedMeasurement);
            possibleSampledBernoulliLikelihood = zeros(numPossibleSampledBernoulli+1,1);
            possibleSampledBernoulliLikelihood(end) = 1;
            possibleSampledBernoulli = cell(numPossibleSampledBernoulli,1);
            possibleNewBernoulli = cell(numPossibleSampledBernoulli,1);
            possibleRemoveIndex = zeros(numPossibleSampledBernoulli,1);
            for j = 1:numPossibleSampledBernoulli
                measurementAssociation(1) = unusedMeasurement(j);
                %perform Bernoulli filtering
                if ellipsoidalGatingforBernoulliPrecompute(numTrajectoryMixture,predictedMean,inverseInnovationCovariance,currentMeasurements,measurementAssociation,gateSize)
                    [updatedBernoulli,isValid] = BernoulliFiltering(pastBernoulli,multiObjectDynamicModel,multiObjectMeasurementModel,measurements,measurementAssociation,timeStep-1,minimumBirthTimeProbability,minimumEndTimeProbability,gateSize);
                else
                    isValid = false;
                end
                %initiate a new Bernoulli if possible for previous measurement assignment
                initiateNewBernoulli = false;
                if isValid && currentMeasurementAssociation > 0 && measurementAssociatedToNew{timeStep}(currentMeasurementAssociation)
                    initiateNewBernoulli = true;
                    newBernoulli = newlyDetectedTrajectory{timeStep}{currentMeasurementAssociation};
                end
                if isValid
                    %check if sampled measurement is used for initiating a new Bernoulli
                    removeNewBernoulli = false;
                    index = find(newDetection==measurementAssociation(1));
                    if ~isempty(index)
                        %delete corresponding Bernoulli
                        removeNewBernoulli = true;
                        indexBernoulliRemove = indexNewlyDetectedBernoulli(index);
                    end

                    changedLikelihood = updatedBernoulli.likelihood - sampleMultiBernoulli(i).likelihood;
                    if currentMeasurementAssociation > 0
                        if initiateNewBernoulli
                            changedLikelihood = changedLikelihood + newBernoulli.likelihood;
                            possibleNewBernoulli{j} = newBernoulli;
                        else
                            changedLikelihood = changedLikelihood + log(multiObjectMeasurementModel.PoissonClutterIntensity);
                        end
                    end
                    if measurementAssociation(1) > 0
                        if removeNewBernoulli
                            possibleRemoveIndex(j) = indexBernoulliRemove;
                            changedLikelihood = changedLikelihood - sampleMultiBernoulli(indexBernoulliRemove).likelihood;
                        else
                            changedLikelihood = changedLikelihood - log(multiObjectMeasurementModel.PoissonClutterIntensity);
                        end
                    end
                    possibleSampledBernoulliLikelihood(j) = exp(changedLikelihood);
                    possibleSampledBernoulli{j} = updatedBernoulli;
                end
            end
            %sample an update action
            sampleIndex = find(rand < cumsum(possibleSampledBernoulliLikelihood/sum(possibleSampledBernoulliLikelihood)),1);
            if sampleIndex < numPossibleSampledBernoulli+1
                sampleMetropolisHastingsLikelihoodDelta(iter) = sampleMetropolisHastingsLikelihoodDelta(iter) + log(possibleSampledBernoulliLikelihood(sampleIndex));
                sampleMultiBernoulli(i) = possibleSampledBernoulli{sampleIndex};
                if ~isempty(possibleNewBernoulli{sampleIndex})
                    newMultiBernoulli = [newMultiBernoulli possibleNewBernoulli{sampleIndex}];
                end
                if possibleRemoveIndex(sampleIndex) > 0
                    indexRemove = [indexRemove possibleRemoveIndex(sampleIndex)];
                end
            end
            sampleMultiBernoulli(indexRemove) = [];
            if ~isempty(newMultiBernoulli)
                sampleMultiBernoulli = [sampleMultiBernoulli newMultiBernoulli];
            end
            %split action
        case 2
            %sample a Bernoulli
            i = samplingPool(randi(numSampleBernoulli,1));
            firstDetectedTimeStep = sampleMultiBernoulli(i).firstDetectedTimeStep;
            associationHistory = sampleMultiBernoulli(i).associationHistory;
            detectedTimeStep = find(associationHistory>0);
            lastDetectedTimeStep = detectedTimeStep(end);

            performSplit = false;
            %sample a time step
            splitTimeStep = detectedTimeStep(randi(length(detectedTimeStep)-1,1)+1);
            splitMeasurementAssociation = associationHistory(splitTimeStep);

            %check if a new Bernoulli can be initiated at this time step
            if measurementAssociatedToNew{splitTimeStep}(splitMeasurementAssociation)
                %construct new Bernoulli
                if splitTimeStep == lastDetectedTimeStep
                    newBernoulli = newlyDetectedTrajectory{splitTimeStep}{splitMeasurementAssociation};
                    performSplit = true;
                else
                    measurementAssociation = associationHistory(splitTimeStep+1:end);
                    initialBernoulli = newlyDetectedTrajectory{splitTimeStep}{splitMeasurementAssociation}.past{splitTimeStep};
                    initialBernoulli.past = newlyDetectedTrajectory{splitTimeStep}{splitMeasurementAssociation}.past;
                    if ellipsoidalGatingforBernoulli(initialBernoulli,multiObjectDynamicModel,multiObjectMeasurementModel,measurements,measurementAssociation,splitTimeStep,gateSize)
                        [newBernoulli,isValid] = BernoulliFiltering(initialBernoulli,multiObjectDynamicModel,multiObjectMeasurementModel,measurements,measurementAssociation,splitTimeStep,minimumBirthTimeProbability,minimumEndTimeProbability,gateSize);
                    else
                        isValid = false;
                    end
                    if isValid
                        performSplit = true;
                    end
                end
            end

            %construct the truncated Bernoulli
            if performSplit
                pastBernoulli = sampleMultiBernoulli(i).past{splitTimeStep-1};
                pastBernoulli.past = sampleMultiBernoulli(i).past;
                measurementAssociation = zeros(1,multiObjectDynamicModel.totalTimeSteps-splitTimeStep+1);
                truncatedBernoulli = BernoulliFiltering(pastBernoulli,multiObjectDynamicModel,multiObjectMeasurementModel,measurements,measurementAssociation,splitTimeStep-1,minimumBirthTimeProbability,minimumEndTimeProbability,gateSize);

                %perform sampling
                changedLikelihood = newBernoulli.likelihood + truncatedBernoulli.likelihood - sampleMultiBernoulli(i).likelihood;
                lastDetectedTimeStep = find(truncatedBernoulli.associationHistory>0,1,'last');
                mergeTimeStep1 = find([sampleMultiBernoulli.firstDetectedTimeStep] > lastDetectedTimeStep);
                mergeTimeStep2 = find(arrayfun(@(x) find(x.associationHistory>0,1,'last'), sampleMultiBernoulli) < sampleMultiBernoulli(i).firstDetectedTimeStep);
                mergeTimeStep = [mergeTimeStep1 mergeTimeStep2];
                acceptanceProbability = min(1,exp(changedLikelihood)/(1/numSampleBernoulli/(length(detectedTimeStep)-1))*(1/length(mergeTimeStep)/(numBernoulli+1)));
                if rand < acceptanceProbability
                    sampleMetropolisHastingsLikelihoodDelta(iter) = changedLikelihood;
                    sampleMultiBernoulli(i) = truncatedBernoulli;
                    sampleMultiBernoulli(end+1) = newBernoulli;
                end
            end
            %merge action
        case 3
            %sample two Bernoullis
            performMerge = false;
            i = randi(numBernoulli,1);
            lastDetectedTimeStep = find(sampleMultiBernoulli(i).associationHistory>0,1,'last');
            mergeTimeStep1 = find([sampleMultiBernoulli.firstDetectedTimeStep] > lastDetectedTimeStep);
            mergeTimeStep2 = find(arrayfun(@(x) find(x.associationHistory>0,1,'last'), sampleMultiBernoulli) < sampleMultiBernoulli(i).firstDetectedTimeStep);
            mergeTimeStep = [mergeTimeStep1 mergeTimeStep2];

            if ~isempty(mergeTimeStep)
                firstDetectedTimeStep = sampleMultiBernoulli(i).firstDetectedTimeStep;
                associationHistory = sampleMultiBernoulli(i).associationHistory;
                j = mergeTimeStep(randi(length(mergeTimeStep),1));
                firstDetectedTimeStepPrime = sampleMultiBernoulli(j).firstDetectedTimeStep;
                associationHistoryPrime = sampleMultiBernoulli(j).associationHistory;
                %append Bernoulli j to Bernoulli i
                if firstDetectedTimeStepPrime > lastDetectedTimeStep
                    pastBernoulli = sampleMultiBernoulli(i).past{firstDetectedTimeStepPrime-1};
                    if pastBernoulli.isAlive
                        pastBernoulli.past = sampleMultiBernoulli(i).past;
                        measurementAssociation = associationHistoryPrime(firstDetectedTimeStepPrime:end);
                        if ellipsoidalGatingforBernoulli(pastBernoulli,multiObjectDynamicModel,multiObjectMeasurementModel,measurements,measurementAssociation,firstDetectedTimeStepPrime-1,gateSize)
                            [mergedBernoulli,isValid] = BernoulliFiltering(pastBernoulli,multiObjectDynamicModel,multiObjectMeasurementModel,measurements,measurementAssociation,firstDetectedTimeStepPrime-1,minimumBirthTimeProbability,minimumEndTimeProbability,gateSize);
                        else
                            isValid = false;
                        end
                        if isValid
                            performMerge = true;
                        end
                    end
                    %append Bernoulli i to Bernoulli j
                else
                    pastBernoulli = sampleMultiBernoulli(j).past{firstDetectedTimeStep-1};
                    if pastBernoulli.isAlive
                        pastBernoulli.past = sampleMultiBernoulli(j).past;
                        measurementAssociation = associationHistory(firstDetectedTimeStep:end);
                        if ellipsoidalGatingforBernoulli(pastBernoulli,multiObjectDynamicModel,multiObjectMeasurementModel,measurements,measurementAssociation,firstDetectedTimeStep-1,gateSize)
                            [mergedBernoulli,isValid] = BernoulliFiltering(pastBernoulli,multiObjectDynamicModel,multiObjectMeasurementModel,measurements,measurementAssociation,firstDetectedTimeStep-1,minimumBirthTimeProbability,minimumEndTimeProbability,gateSize);
                        else
                            isValid = false;
                        end
                        if isValid
                            performMerge = true;
                        end
                    end
                end
            end

            %perform sampling
            if performMerge
                changedLikelihood = mergedBernoulli.likelihood - sampleMultiBernoulli(i).likelihood - sampleMultiBernoulli(j).likelihood;
                detectedTimeStep = find(mergedBernoulli.associationHistory>0);
                acceptanceProbability = min(1,exp(changedLikelihood)/(1/length(mergeTimeStep)/numBernoulli)*(1/(numSampleBernoulli+1)/(length(detectedTimeStep)-1)));
                if rand < acceptanceProbability
                    sampleMetropolisHastingsLikelihoodDelta(iter) = changedLikelihood;
                    sampleMultiBernoulli(i) = mergedBernoulli;
                    sampleMultiBernoulli(j) = [];
                end
            end
            %switch action
        case 4
            %sample two Bernoullis
            sampleIndex = datasample(1:numSampleBernoulli,2,'Replace',false);
            i = samplingPool(sampleIndex(1));
            j = samplingPool(sampleIndex(2));
            firstDetectedTimeStep = sampleMultiBernoulli(i).firstDetectedTimeStep;
            associationHistory = sampleMultiBernoulli(i).associationHistory;
            lastDetectedTimeStep = find(associationHistory>0,1,'last');
            firstDetectedTimeStepPrime = sampleMultiBernoulli(j).firstDetectedTimeStep;
            associationHistoryPrime = sampleMultiBernoulli(j).associationHistory;
            lastDetectedTimeStepPrime = find(associationHistoryPrime>0,1,'last');

            timePeriod = max(firstDetectedTimeStep,firstDetectedTimeStepPrime):max(lastDetectedTimeStep,lastDetectedTimeStepPrime);
            if ~isempty(timePeriod) && length(timePeriod) > 1
                timeStep = randi([timePeriod(1)+1,timePeriod(end)],1);
                performSwitch = false;
                %swap the measurement association of Bernoulli i and Bernoulli j since this time step
                pastBernoulli = sampleMultiBernoulli(i).past{timeStep-1};
                pastBernoulli.past = sampleMultiBernoulli(i).past;
                measurementAssociation = associationHistoryPrime(timeStep:end);
                if ellipsoidalGatingforBernoulli(pastBernoulli,multiObjectDynamicModel,multiObjectMeasurementModel,measurements,measurementAssociation,timeStep-1,gateSize)
                    [switchedBernoulli,isValid] = BernoulliFiltering(pastBernoulli,multiObjectDynamicModel,multiObjectMeasurementModel,measurements,measurementAssociation,timeStep-1,minimumBirthTimeProbability,minimumEndTimeProbability,gateSize);
                else
                    isValid = false;
                end
                if isValid
                    pastBernoulli = sampleMultiBernoulli(j).past{timeStep-1};
                    pastBernoulli.past = sampleMultiBernoulli(j).past;
                    measurementAssociation = associationHistory(timeStep:end);
                    if ellipsoidalGatingforBernoulli(pastBernoulli,multiObjectDynamicModel,multiObjectMeasurementModel,measurements,measurementAssociation,timeStep-1,gateSize)
                        [switchedBernoulliPrime,isValidPrime] = BernoulliFiltering(pastBernoulli,multiObjectDynamicModel,multiObjectMeasurementModel,measurements,measurementAssociation,timeStep-1,minimumBirthTimeProbability,minimumEndTimeProbability,gateSize);
                    else
                        isValidPrime = false;
                    end
                    if isValidPrime
                        performSwitch = true;
                    end
                end

                %perform sampling
                if performSwitch
                    changedLikelihood = switchedBernoulli.likelihood + switchedBernoulliPrime.likelihood - sampleMultiBernoulli(i).likelihood - sampleMultiBernoulli(j).likelihood;
                    acceptanceProbability = min(1,exp(changedLikelihood));
                    if rand < acceptanceProbability
                        sampleMetropolisHastingsLikelihoodDelta(iter) = sampleMetropolisHastingsLikelihoodDelta(iter) + changedLikelihood;
                        sampleMultiBernoulli(i) = switchedBernoulli;
                        sampleMultiBernoulli(j) = switchedBernoulliPrime;
                    end
                end
            end
    end

    newIterLikelihood = sum(sampleMetropolisHastingsLikelihoodDelta);
    if newIterLikelihood > highestLikelihood
        highestLikelihood = newIterLikelihood;
        mostLikelyMultiBernoulli =  sampleMultiBernoulli;
    end
end
fprintf('\nFinished\n');

objectTrajectoryBatchEstimate = [];
for i = 1:length(mostLikelyMultiBernoulli)
    localHypothesis = mostLikelyMultiBernoulli(i);
    if localHypothesis.existenceProbability >= estimateExistenceProbabilityThreshold

        initialBernoulli = newlyDetectedTrajectoryOriginal{localHypothesis.firstDetectedTimeStep}{localHypothesis.associationHistory(localHypothesis.firstDetectedTimeStep)};
        measurementAssociation = localHypothesis.associationHistory(localHypothesis.firstDetectedTimeStep+1:end);
        Bernoulli = BernoulliFilteringSimplified(initialBernoulli,multiObjectDynamicModel,multiObjectMeasurementModel,measurements,measurementAssociation,localHypothesis.firstDetectedTimeStep,minimumEndTimeProbability);

        [~,birthTimeIndex] = max(Bernoulli.birthTimeProbability);
        trajectoryComponent = Bernoulli.trajectoryMixture(birthTimeIndex);

        [~,endTime] = max(Bernoulli.endTimeProbability(birthTimeIndex,:));
        stateSequence = RTSsmoothing(trajectoryComponent.filterMean(:,Bernoulli.birthTime(birthTimeIndex):endTime),trajectoryComponent.filterCovariance(:,:,Bernoulli.birthTime(birthTimeIndex):endTime),multiObjectDynamicModel);

        objectTrajectoryBatchEstimate(end+1).birthTime = Bernoulli.birthTime(birthTimeIndex);
        objectTrajectoryBatchEstimate(end).endTime = endTime;
        objectTrajectoryBatchEstimate(end).stateSequence = stateSequence;
        objectTrajectoryBatchEstimate(end).measurementSequence = localHypothesis.associationHistory(Bernoulli.birthTime(birthTimeIndex):endTime);
    end
end

%evaluating smoothing performance after batch processing
[tgospa_cost_mini_batch, loc_cost_mini_batch, miss_cost_mini_batch, fa_cost_mini_batch, switch_cost_mini_batch] = ...
    performanceEvaluation(objectTrajectory,objectTrajectoryBatchEstimate,multiObjectDynamicModel.totalTimeSteps,TGOSPA);

showResults(objectTrajectory,objectTrajectoryBatchEstimate,multiObjectMeasurementModel)

