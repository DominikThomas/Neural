package io.github.dominikthomas.neuralnet;

import java.util.Arrays;

import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;
import org.springframework.util.StopWatch;

import io.github.dominikthomas.neuralnet.beans.NeuralException;
import io.github.dominikthomas.neuralnet.beans.NeuralNet;
import io.github.dominikthomas.neuralnet.math.RandomNumberGenerator;

/**
 *
 * NeuralNetConsoleTest
 * This class is solely used for creating and testing your very first Neural 
 * Network in Java 
 *
 * @author Alan de Souza, Fábio Soares, Dominik Thomas
 * @version 0.2
 */
public class NeuralNetSpring {
	@SuppressWarnings("resource")
	public static void main(String[] args) {
		
		StopWatch stopWatch = new StopWatch();
		stopWatch.start();
 
		// loading the definitions from the given XML file
		ApplicationContext context = new ClassPathXmlApplicationContext(
				"neuralnet/applicationContext.xml");

		RandomNumberGenerator.setSeed(System.currentTimeMillis());
 
		try {
			NeuralNet nn = (NeuralNet) context
					.getBean("neuralNet");
			
			stopWatch.stop();
	        System.out.println("Init took: " + stopWatch.getLastTaskTimeMillis() + "ms.");
	        stopWatch.start();
	        
			nn.init();
	        
	        double [] neuralInput = { 1.5 , 0.5 , 2.5};
	        
	        double [] neuralOutput;
				nn.setInputs(neuralInput);
	        nn.calc();
	        neuralOutput=nn.getOutputs();
	        System.out.println(Arrays.toString(neuralOutput));
	        
	        stopWatch.stop();        
	        System.out.println("Elapsed time: " + stopWatch.getLastTaskTimeMillis() + "ms.");
	        stopWatch.start();
	        
	        nn.init();
	        
	        neuralInput[0] = 1.0;
	        neuralInput[1] = 2.1;
	        neuralInput[2] = 0.1;
	        
	        nn.setInputs(neuralInput);
	        nn.calc();
	        neuralOutput=nn.getOutputs();
	        System.out.println(Arrays.toString(neuralOutput));
		} catch (NeuralException e) {
			e.printStackTrace();
		}
        
        stopWatch.stop();        
        System.out.println("Elapsed time: " + stopWatch.getLastTaskTimeMillis() + "ms.");
        
    }
}
