package com.weka;

import java.util.ArrayList;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class ProgrammaticUse {

	public static void main(String[] args) throws Exception {
		// / Step 1: Express the problem with features
		// / This step corresponds to the engineering task needed to write an .arff file.
		// / Let’s put all our features in a weka.core.FastVector.
		// / Each feature is contained in a weka.core.Attribute object.

		// Declare two numeric attributes
		Attribute Attribute1 = new Attribute("firstNumeric");
		Attribute Attribute2 = new Attribute("secondNumeric");

		// Declare a nominal attribute along with its values
		ArrayList<String> fvNominalVal = new ArrayList<String>(3);
		fvNominalVal.add("blue");
		fvNominalVal.add("gray");
		fvNominalVal.add("black");
		Attribute Attribute3 = new Attribute("aNominal", fvNominalVal);

		// Declare the class attribute along with its values
		ArrayList<String> fvClassVal = new ArrayList<String>(2);
		fvClassVal.add("positive");
		fvClassVal.add("negative");
		Attribute ClassAttribute = new Attribute("theClass", fvClassVal);

		// Declare the feature vector
		ArrayList<Attribute> fvWekaAttributes = new ArrayList<Attribute>(4);
		fvWekaAttributes.add(Attribute1);
		fvWekaAttributes.add(Attribute2);
		fvWekaAttributes.add(Attribute3);
		fvWekaAttributes.add(ClassAttribute);
		System.out.println(fvWekaAttributes);

		// / Step 2: Train a Classifier
		// / Training requires 1) having a training set of instances and 2) choosing a classifier.
		// / Let’s first create an empty training set (weka.core.Instances).
		// / We named the relation “Rel”.
		// / The attribute prototype is declared using the vector from step 1.
		// / We give an initial set capacity of 10.
		// / We also declare that the class attribute is the fourth one in the vector (see step 1)

		// Create an empty training set
		Instances isTrainingSet = new Instances("Rel", fvWekaAttributes, 10);
		// Set class index
		isTrainingSet.setClassIndex(3);

		// / Now, let’s fill the training set with one instance (weka.core.Instance):
		// / Create the instance
		Instance iExample = new DenseInstance(4);
		iExample.setValue((Attribute) fvWekaAttributes.get(0), 1.0);
		iExample.setValue((Attribute) fvWekaAttributes.get(1), 0.5);
		iExample.setValue((Attribute) fvWekaAttributes.get(2), "gray");
		iExample.setValue((Attribute) fvWekaAttributes.get(3), "positive");

		// add the instance
		isTrainingSet.add(iExample);

		// / Finally, Choose a classifier (weka.classifiers.Classifier) and create the model. Let’s, for example, create
		// a naive Bayes classifier (weka.classifiers.bayes.NaiveBayes)
		// / Create a naïve bayes classifier
		Classifier cModel = (Classifier) new NaiveBayes();
		cModel.buildClassifier(isTrainingSet);

		// / Step 4: use the classifier
		// / For real world applications, the actual use of the classifier is the ultimate goal. Here’s the simplest way
		// to achieve that. Let’s say we’ve built an instance (named iUse) as explained in step 2:
		// / Specify that the instance belong to the training set
		// in order to inherit from the set description
		Instance iUse = iExample;
		iUse.setDataset(isTrainingSet);

		// Get the likelihood of each classes
		// fDistribution[0] is the probability of being “positive”
		// fDistribution[1] is the probability of being “negative”
		double[] fDistribution = cModel.distributionForInstance(iUse);
		for (int i = 0; i < fDistribution.length; i++) {
			System.out.println(fDistribution[i]);
		}
	}

}
