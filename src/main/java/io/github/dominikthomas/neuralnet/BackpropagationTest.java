package io.github.dominikthomas.neuralnet;

import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;

import io.github.dominikthomas.neuralnet.beans.NeuralException;
import io.github.dominikthomas.neuralnet.learn.Backpropagation;
import io.github.dominikthomas.neuralnet.math.RandomNumberGenerator;

public class BackpropagationTest {
    public static void main(String[] args){
    	
		// loading the definitions from the given XML file
		@SuppressWarnings("resource")
		ApplicationContext context = new ClassPathXmlApplicationContext(
				"backpropagationtest/applicationContext.xml");

		RandomNumberGenerator.setSeed(System.currentTimeMillis());

		System.out.println("Creating Neural Network...");
		
		Backpropagation backpropagation = (Backpropagation) context.getBean("backpropagation");
		backpropagation.init();
        
        System.out.println("Neural Network created!");
        backpropagation.getNeuralNet().print();
        
        backpropagation.getTrainingDataSet().printInput();
        backpropagation.getTrainingDataSet().printTargetOutput();
        
        System.out.println("Getting the first output of the neural network");
        
        
        try{
            backpropagation.forward();
            backpropagation.getTrainingDataSet().printNeuralOutput();
            
            backpropagation.train();
            System.out.println("End of training");
            if(backpropagation.getMinOverallError()>=backpropagation.getOverallGeneralError()){
                System.out.println("Training successful!");
            }
            else{
                System.out.println("Training was unsuccessful");
            }
            System.out.println("Overall Error:"
                        +String.valueOf(backpropagation.getOverallGeneralError()));
            System.out.println("Min Overall Error:"
                        +String.valueOf(backpropagation.getMinOverallError()));
            System.out.println("Epochs of training:"
                        +String.valueOf(backpropagation.getEpoch()));
            
            System.out.println("Target Outputs:");
            backpropagation.getTrainingDataSet().printTargetOutput();
            
            System.out.println("Neural Output after training:");
            backpropagation.forward();
            backpropagation.getTrainingDataSet().printNeuralOutput();
        }
        catch(NeuralException ne){
            
        }

    }
}
