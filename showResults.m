function showResults(objectTrajectory,objectTrajectoryEstimate,multiObjectMeasurementModel)

figure
hold on
for i = 1:length(objectTrajectory)
    locationSequence = multiObjectMeasurementModel.observationMatrix*objectTrajectory(i).stateSequence;
    plot(locationSequence(1,1),locationSequence(2,1),'ko','MarkerSize',8,'LineWidth',1.5)
    plot(locationSequence(1,:),locationSequence(2,:),'k','LineWidth',1.5)
end

for i = 1:length(objectTrajectoryEstimate)
    locationSequence = multiObjectMeasurementModel.observationMatrix*objectTrajectoryEstimate(i).stateSequence;
    plot(locationSequence(1,1),locationSequence(2,1),'bo','MarkerSize',8,'LineWidth',1.5)
    plot(locationSequence(1,:),locationSequence(2,:),'b','LineWidth',1.5)
end

xlim(multiObjectMeasurementModel.rangeOfInterest(1,:))
ylim(multiObjectMeasurementModel.rangeOfInterest(2,:))

xlabel('x-position (m)')
ylabel('y-position (m)')

end