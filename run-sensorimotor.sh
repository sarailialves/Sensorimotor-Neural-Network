#!/bin/bash

RED='\033[0;35m'
NC='\033[0m' # No Color
BLUE='\033[1;34m'
LBLUE='\033[0;36m'

echo -e "${RED}Creating necessary directories...${NC}"

mkdir sfield
mkdir mfield
mkdir pfield
mkdir pickle

echo -e "${RED}Installing necessary Python packages...${NC}"

pip install numpy matplotlib scipy Pillow sklearn tqdm

echo -e "${BLUE}Starting the application...${NC}"

echo -e "${BLUE}Step 1/5: Defining the dataset${NC}"

echo -e "${LBLUE}Enter \"1\" for translational dataset or \"2\" for rotational/zoom dataset:${NC}"
read varname

if [ "$varname" = 1 ]
then
	python dataset1.py
fi

if [ "$varname" = 2 ]
then
	python dataset2.py
fi

echo -e "${BLUE}Step 2/5: Learning the one layer network${NC}"
python learner.py

echo -e "${BLUE}Step 3/5: Factorizing the one layer network${NC}"
python factor.py

echo -e "${BLUE}Step 4/5: Learning the first part of the Sensorimotor network${NC}"
python main_fast.py

echo -e "${BLUE}Step 5/5: Learning the second part of the Sensorimotor network${NC}"

if [ "$varname" = 1 ]
then
	python main2_fast_1.py
fi

if [ "$varname" = 2 ]
then
	python main2_fast_2.py
fi

read -n1 -p "Press any key to exit."
exit