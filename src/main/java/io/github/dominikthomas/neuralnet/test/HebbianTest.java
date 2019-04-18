package io.github.dominikthomas.neuralnet.test;

import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;
import org.springframework.util.StopWatch;

import io.github.dominikthomas.neuralnet.beans.NeuralException;
import io.github.dominikthomas.neuralnet.data.NeuralDataSet;
import io.github.dominikthomas.neuralnet.learn.Hebbian;

public class HebbianTest {
    public static void main(String[] args){
        
		StopWatch stopWatch = new StopWatch();
		stopWatch.start();
 
		System.out.println("Creating Neural Network...");
		// loading the definitions from the given XML file
		@SuppressWarnings("resource")
		ApplicationContext context = new ClassPathXmlApplicationContext(
				"hebbiantest/applicationContext.xml");

		Hebbian hebbian = (Hebbian) context.getBean("hebbian");
		hebbian.init();
		
        System.out.println("Neural Network created!");
        
        
        NeuralDataSet neuralDataSet = (NeuralDataSet) context.getBean("trainingDataSet");        
        System.out.println("Dataset created");
        neuralDataSet.printInput();
        
        System.out.println("Getting the first output of the neural network");
        
        try{
            hebbian.forward();
            neuralDataSet.printNeuralOutput();
        
            System.out.println("Beginning training");
            
            hebbian.train();
            
            System.out.println("End of training");
            System.out.println("Epochs of training:"
                        +String.valueOf(hebbian.getEpoch()));
            
            System.out.println("Neural Output after training:");
            hebbian.forward();
            neuralDataSet.printNeuralOutput();
            
        }
        catch(NeuralException ne){
            
        }
        
        
    }    
    
}
