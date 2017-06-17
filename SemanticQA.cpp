#include "Import.hpp"
#include "File.hpp"
#include "Model.hpp"
#include "SemanticModel.hpp"
#include "DetailedConfig.hpp"
#include "Task.hpp"
#include "DataProcess.hpp"
#include <boost/format.hpp>  
#include <boost/tokenizer.hpp>  
#include <boost/algorithm/string.hpp>  

int main(int argc, char* argv[])
{
	Model * model = nullptr;
	AscentAdaDeltaPositive solver(0.6, 1e-6);

	model = new JointMFactor(SimpleQuestion, 10, 10, 2.5, 0.04, solver);
	model->train(500, [=](int epos)
	{
		string base_url = "H:\\Model\\QA\\";
		model->save(""
	});

	return 0;
}