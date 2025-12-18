package nn.examples;

import nn.core.Network;
import nn.activation.*;
import nn.initialization.*;
import nn.loss.*;
import nn.training.Trainer;
import nn.data.DataUtils;

public class SimpleExample {
    
    public static void main(String[] args) {
        System.out.println("=== Simple Neural Network Example ===");
        System.out.println();
        
        double[][] inputs = {
            {0.0, 0.0},
            {0.0, 1.0},
            {1.0, 0.0},
            {1.0, 1.0}
        };
        
        double[][] outputs = {
            {0.0},
            {1.0},
            {1.0},
            {0.0}
        };
        
        Network network = new Network(2);
        network.addLayer(4, new Sigmoid());
        network.addLayer(1, new Sigmoid());
        
        network.setLossFunction(new MSE());
        
        network.setWeightInitializer(new Xavier());
        
        Trainer trainer = new Trainer(network);
        trainer.setLearningRate(0.5);
        trainer.setEpochs(1000);
        trainer.setBatchSize(4);
        
        System.out.println("Training network on XOR problem...");
        trainer.train(inputs, outputs);
        
        System.out.println("\nPredictions:");
        for (int i = 0; i < inputs.length; i++) {
            double[] prediction = network.predict(inputs[i]);
            System.out.printf("Input: [%.1f, %.1f] -> Output: %.3f (Expected: %.1f)%n",
                inputs[i][0], inputs[i][1], prediction[0], outputs[i][0]);
        }
    }
}


