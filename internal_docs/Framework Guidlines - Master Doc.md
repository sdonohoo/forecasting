# Benchmarking Guide – TOC 
## Overview - framework doc
## Framework vision and goals
## Definitions
	a. Overview of terms
	b. Framework goals
	c. Dataset
	d. Problem
	e. Benchmark
	f. Benchmark backlog
	g. Implementation
## New Benchmark Implementation Guideline  
	a. Motivation (why this problem is interesting)
	b. Requirements for solving the problem
	c. Assumptions (what can be assumed by the submitter)
	d. Quality threshold requirements  (Accuracy requirements)
	e. Other threshold requirements (if any)
	f. Data Definition - problem doc
	    i. General description of the data set and its location
	    ii. Training/validation/testing guidelines 
			iii. Pre-processing guidelines (if any)
	g. Modeling requirements - problem doc
		i. Requirements in respect to modeling approach (if any)
		ii. Requirements  for algorithm usage (if any)
		iii. Requirements for hyper parameters tuning (if any)
		iv. Feature extraction guideline (if any)
	h. Reporting guideline (how to report the results
##  Reproduction - framework doc, problem doc
	a. Requirements for reproduction
		i. Quality requirements – allowed range of quality metric
		ii. Simplicity – maximum time for deployment
		iii. Dealing with reproduction errors and issue 
	b. Required reproduction instruction and artifacts (that the submitter must provide during submission)
		i. Scripts 
		ii. Docker images 
		iii. Other
	c. Available reproduction tools available by the framework
	d. Required range of time by which the benchmark should be reproducible
	e. Requirement for supporting reproducibility over time 
## Implementation Reviewing Guide - framework doc
	a. Overview
	b. Description of the reviewing process
	c. Guidelines for accepting/approving a benchmark
	d. Guideline for rejecting a benchmark
	e. Guidelines for updating results board
