package com.weka;

import javax.crypto.Cipher;
import javax.crypto.CipherInputStream;

import weka.classifiers.trees.J48;

/**
 * Hello world!
 *
 */
public class App {
	public static void main(String[] args) {
		System.out.println("Hello World!");

		App readVersion = new App();
		readVersion.readVersionInfoInManifest();
	}

	public void readVersionInfoInManifest() {

		// build an object whose class is in the target jar
		J48 object = new J48();

		// navigate from its class object to a package object
		Package objPackage = object.getClass().getPackage();

		// examine the package object
		String name = objPackage.getSpecificationTitle();
		String version = objPackage.getSpecificationVersion();
		// some jars may use 'Implementation Version' entries in the manifest instead

		System.out.println("Package name: " + name);
		System.out.println("Package version: " + version);
		System.out.println(objPackage);
	}

}
