#pragma once
#include "Import.hpp"
#include "Logging.hpp"

class AscentAdaDeltaPositive
{
protected:
	const double moment;
	const double regularization;

public:
	AscentAdaDeltaPositive(double moment, double regularization)
		:moment(moment), regularization(regularization)
	{
		;
	}

public:
	void logout_parameter() const
	{
		logout.record() << "[Solver]\tAdaDelta";
		logout.record() << "[Setting]\tMoment = " << moment;
		logout.record() << "[Setting]\tRegularization = " << regularization;
	}

public:
	void gradient(double& derv_grad, double& derv_x, double& elem, double& grad) const
	{
		derv_grad = moment * derv_grad + (1 - moment) * grad * grad;
		double derv_elem = sqrt(derv_x + regularization) / sqrt(derv_grad + regularization) * grad;
		derv_x = moment * derv_x + (1 - moment) * derv_elem * derv_elem;

#pragma omp critical
		elem += derv_elem;
	}

	void gradient(vec& derv_grad, vec& derv_x, vec& elem, vec& grad) const
	{
		derv_grad = moment * derv_grad + (1 - moment) * grad % grad;
		vec derv_elem = sqrt(derv_x + regularization) / sqrt(derv_grad + regularization) % grad;
		derv_x = moment * derv_x + (1 - moment) * derv_elem % derv_elem;

#pragma omp critical
		{
			elem += abs(derv_elem) + 1e-10;
			elem = normalise(elem, 2);
		}
	}

	void gradient(mat& derv_grad, mat& derv_x, mat& elem, mat& grad) const
	{
		derv_grad = moment * derv_grad + (1 - moment) * grad % grad;
		mat derv_elem = sqrt(derv_x + regularization) / sqrt(derv_grad + regularization) % grad;
		derv_x = moment * derv_x + (1 - moment) * derv_elem % derv_elem;

#pragma omp critical
		{
			elem += abs(derv_elem) + 1e-10;
			elem = normalise(elem, 2);
		}
	}
};

class DecentAdaDelta
{
protected:
	const double moment;
	const double regularization;

public:
	DecentAdaDelta(double moment, double regularization)
		:moment(moment), regularization(regularization)
	{
		;
	}

public:
	void logout_parameter() const
	{
		logout.record() << "[Solver]\tAdaDelta";
		logout.record() << "[Setting]\tMoment = " << moment;
		logout.record() << "[Setting]\tRegularization = " << regularization;
	}

public:
	void gradient(double& derv_grad, double& derv_x, double& elem, double& grad) const
	{
		derv_grad = moment * derv_grad + (1 - moment) * grad * grad;
		double derv_elem = sqrt(derv_x + regularization) / sqrt(derv_grad + regularization) * grad;
		derv_x = moment * derv_x + (1 - moment) * derv_elem * derv_elem;

		elem -= derv_elem;
	}

	void gradient(vec& derv_grad, vec& derv_x, vec& elem, vec& grad) const
	{
		derv_grad = moment * derv_grad + (1 - moment) * grad % grad;
		vec derv_elem = sqrt(derv_x + regularization) / sqrt(derv_grad + regularization) % grad;
		derv_x = moment * derv_x + (1 - moment) * derv_elem % derv_elem;

		elem -= derv_elem;
	}

	void gradient(mat& derv_grad, mat& derv_x, mat& elem, mat& grad) const
	{
		derv_grad = moment * derv_grad + (1 - moment) * grad % grad;
		mat derv_elem = sqrt(derv_x + regularization) / sqrt(derv_grad + regularization) % grad;
		derv_x = moment * derv_x + (1 - moment) * derv_elem % derv_elem;

		elem -= derv_elem;
	}
};