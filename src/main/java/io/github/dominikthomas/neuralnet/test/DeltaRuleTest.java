package io.github.dominikthomas.neuralnet.test;

import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;
import org.springframework.util.StopWatch;

import io.github.dominikthomas.neuralnet.beans.NeuralException;
import io.github.dominikthomas.neuralnet.learn.DeltaRule;
import io.github.dominikthomas.neuralnet.math.RandomNumberGenerator;

public class DeltaRuleTest {
	@SuppressWarnings("resource")
	public static void main(String[] args) {
		
		StopWatch stopWatch = new StopWatch();
		stopWatch.start();
 
		// loading the definitions from the given XML file
		ApplicationContext context = new ClassPathXmlApplicationContext(
				"deltaruletest/applicationContext.xml");

		RandomNumberGenerator.setSeed(System.currentTimeMillis());
		
		DeltaRule deltaRule = (DeltaRule) context.getBean("deltaRule");
		deltaRule.init();
		
		System.out.println("Dataset created");
		
		deltaRule.getTrainingDataSet().printInput();
		deltaRule.getTrainingDataSet().printTargetOutput();
        
        System.out.println("Getting the first output of the neural network");
        
        try{
            deltaRule.forward();
            deltaRule.getTrainingDataSet().printNeuralOutput();
            
            Double weight = deltaRule.getNeuralNet().getOutputLayer().getWeight(0, 0);
            Double bias = deltaRule.getNeuralNet().getOutputLayer().getWeight(1, 0);
            
            System.out.println("Initial weight:"+String.valueOf(weight));
            System.out.println("Initial bias:"+String.valueOf(bias));
        
            System.out.println("Beginning training");
            
            deltaRule.train();
            
            System.out.println("End of training");
            if(deltaRule.getMinOverallError()>=deltaRule.getOverallGeneralError()){
                System.out.println("Training succesful!");
            }
            else{
                System.out.println("Training was unsuccesful");
            }
            System.out.println("Overall Error:"
                        +String.valueOf(deltaRule.getOverallGeneralError()));
            System.out.println("Min Overall Error:"
                        +String.valueOf(deltaRule.getMinOverallError()));
            System.out.println("Epochs of training:"
                        +String.valueOf(deltaRule.getEpoch()));
            
            System.out.println("Target Outputs:");
            deltaRule.getTrainingDataSet().printTargetOutput();
            
            System.out.println("Neural Output after training:");
            deltaRule.forward();
            deltaRule.getTrainingDataSet().printNeuralOutput();
            
            weight = deltaRule.getNeuralNet().getOutputLayer().getWeight(0, 0);
            bias = deltaRule.getNeuralNet().getOutputLayer().getWeight(1, 0);
            
            System.out.println("Weight found:"+String.valueOf(weight));
            System.out.println("Bias found:"+String.valueOf(bias));
            
    		deltaRule.getTestingDataSet().printInput();
    		deltaRule.getTestingDataSet().printTargetOutput();

            deltaRule.test();
            deltaRule.getTestingDataSet().printNeuralOutput();
        }
        catch(NeuralException ne){
            
        }
        
    }
	
    public static double fncTest(double x){
        return 0.11*x;
    }
    
}