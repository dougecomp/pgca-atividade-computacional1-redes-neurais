/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package exemplos;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import listeners.NeuralNetworkValidationListener;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.util.TransferFunctionType;

/**
 *
 * @author douglas
 */
public class ExercicioComputacional1 {

    public static DataSet getDataSet(String filename, String separator) throws IOException {
        
        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));
                                    
        double[] x = new double[1];
        double[] y = new double[1];

        DataSet trainingSet = new DataSet(1, 1);
        // Ler o conjunto de dados de um arquivo
        
        String line = reader.readLine();
        while(line!=null){
            String[] partes = line.split(separator);
            
            x[0] = Double.parseDouble(partes[0]);
            
            y[0] = Double.parseDouble(partes[1]);

            trainingSet.addRow(x, y);

            line = reader.readLine();
        }   
        reader.close();
        
        return trainingSet;
        
    }
    
    public static void createNnet() {
        
        int qtdInputNeurons = 1;
        int qtdHiddenNeuros = 5;
        int qtdOutputNeurons = 1;
        
        NeuralNetwork mlp = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, qtdInputNeurons, qtdHiddenNeuros, qtdOutputNeurons);
        mlp.addListener(new NeuralNetworkValidationListener());
        
    }
    
    public static void trainNnet() {
        
    }
    
    public static void validateNnet() {
        
    }
    
    public static void testNnet() {
        
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException {
        // TODO code application logic here
        
    }
    
}
