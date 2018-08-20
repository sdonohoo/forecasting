# Pull Request Process

## Overview

The following document is describing the Pull Request Process that we (the framework team) would like to use for the TSPerf development process as well as for the benchmark implementation development. The process applies for both document and code development.

## Terms Definition

* Approver  - A person who is designated to approve PRs
* Developer - A person who develop code or documents
* Pull Request - A request to pull and merge code or documents into the master brunch
* Reviewer - A person who is assigned to review a PR by the developer

## Pull Request PR Process

1. Code/docs development is performs by the developers on a separated brunch outside of the master brunch
2. Upon completion of a given task the developer/submitter will issue a PR that includes the following elements:
   * Code/Doc to be reviewed
   * List of reviewers (at least one reviewer)
   * Designated approver
3. Each of the listed reviewers should review and provide comments for the PR
4. Comments could be of 2 types:
   * General notes that don't require change or update of the submitted code/doc
   * Comment with a request to change the code/doc
5. The designated approver should:
   * Collect all comments and verify implementation
   * Review the entire code/doc for validity
   * Approve the PR (after all comments are processed and completed)
6. After the PR approval, the submitter should merge the relevant code/doc into the master brunch and resolve conflicts (if exist)

## Resource Planing Implications

Since reviewing and approving PRs could be time consuming, it is important to plan and allocating resources in advance for that.Therefor, the following guidelines should be considered:

* At the sprint planing, all expected PRs should be discussed based on inputs from all developers
* At the sprint planning, the designated approver should be designated
* The designated approver should estimate the required effort for reviewing all PRs and allocate the required time for the next sprint accordingly
* A 2nd approver should be assigned in case of possible conflicts or time constraints
* Developers should notify the reviewers in advance at the sprint planning 
* Other developers who take dependency on the PR's code should be included as reviewers
* Reviewers should allocate time for the next sprint for reviewing 


