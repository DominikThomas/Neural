package io.github.dominikthomas.neuralnet;

import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;
import org.springframework.util.StopWatch;

import io.github.dominikthomas.neuralnet.beans.NeuralException;
import io.github.dominikthomas.neuralnet.learn.DeltaRule;

public class AndTest {
	public static void main(String[] args) {
		
		StopWatch stopWatch = new StopWatch();
		stopWatch.start();
 
		// loading the definitions from the given XML file
		@SuppressWarnings("resource")
		ApplicationContext context = new ClassPathXmlApplicationContext(
				"andtest/applicationContext.xml");

		DeltaRule deltaRule = (DeltaRule) context.getBean("deltaRule");
		deltaRule.init();
		
        try{
            deltaRule.forward();
            deltaRule.getTrainingDataSet().printNeuralOutput();
            
            Double weight0 = deltaRule.getNeuralNet().getOutputLayer().getWeight(0, 0);
            Double weight1 = deltaRule.getNeuralNet().getOutputLayer().getWeight(1, 0);
            Double bias = deltaRule.getNeuralNet().getOutputLayer().getWeight(2, 0);
            
            System.out.println("Initial weight0:"+String.valueOf(weight0));
            System.out.println("Initial weight1:"+String.valueOf(weight1));
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
            
            weight0 = deltaRule.getNeuralNet().getOutputLayer().getWeight(0, 0);
            weight1 = deltaRule.getNeuralNet().getOutputLayer().getWeight(1, 0);
            bias = deltaRule.getNeuralNet().getOutputLayer().getWeight(2, 0);
            
            System.out.println("Weight0 found:"+String.valueOf(weight0));
            System.out.println("Weight1 found:"+String.valueOf(weight1));
            System.out.println("Bias found:"+String.valueOf(bias));
            
        }
        catch(NeuralException ne){
            
        }
        
    }
}
