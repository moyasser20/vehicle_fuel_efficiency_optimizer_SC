package case_study;

import ga.core.Chromosome;
import ga.core.GeneticAlgorithm;

public class FuelEfficiencyFitness implements GeneticAlgorithm.FitnessFunction {
    
    @Override
    public double evaluate(Chromosome chromosome) {
        double totalEfficiency = 0.0;
        
        // Each chromosome represents driving pattern: [speed1, gear1, speed2, gear2, ...]
        for (int i = 0; i < chromosome.getLength(); i += 2) {
            if (i + 1 < chromosome.getLength()) {
                double speed = chromosome.getDoubleValue(i);
                double gear = chromosome.getDoubleValue(i + 1);
                
                // Convert to real values
                double actualSpeed = speed * 100.0; // 0-100 km/h
                int actualGear = (int) Math.round(gear * 5 + 1); // 1-6 gears
                
                // Simple fuel efficiency calculation
                double efficiency = calculateEfficiency(actualSpeed, actualGear);
                totalEfficiency += efficiency;
            }
        }
        
        // Return average efficiency
        int steps = chromosome.getLength() / 2;
        return totalEfficiency / steps;
    }
    
    private double calculateEfficiency(double speed, int gear) {
        // Simple model: optimal speed around 60 km/h, higher gears are better
        double speedFactor = 1.0 - Math.abs(speed - 60.0) / 60.0;
        double gearFactor = 0.5 + (gear * 0.1);
        
        return 10.0 * speedFactor * gearFactor; // Base efficiency of 10 km/L
    }
}
