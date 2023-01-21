#pragma once

namespace Octopus
{
    Eigen::Matrix<float,1,Eigen::Dynamic,Eigen::RowMajor> Eigenize(const OctopusRowVectorXf & m)
    {
        Eigen::Matrix<float,1,Eigen::Dynamic,Eigen::RowMajor> r(m.cols());
        for(size_t i = 0; i < m.cols(); i++)        
                r(i) = m(i);
        return r;
    }
    Eigen::Matrix<double,1,Eigen::Dynamic,Eigen::RowMajor> Eigenize(const OctopusRowVectorXd & m)
    {
        Eigen::Matrix<double,1,Eigen::Dynamic,Eigen::RowMajor> r(m.cols());
        for(size_t i = 0; i < m.cols(); i++)        
                r(i) = m(i);
        return r;
    }
    Eigen::Matrix<std::complex<float>,1,Eigen::Dynamic,Eigen::RowMajor> Eigenize(const OctopusRowVectorXcf & m)
    {
        Eigen::Matrix<std::complex<float>,1,Eigen::Dynamic,Eigen::RowMajor> r(m.cols());
        for(size_t i = 0; i < m.cols(); i++)        
                r(i) = m(i);
        return r;
    }
    Eigen::Matrix<std::complex<double>,1,Eigen::Dynamic,Eigen::RowMajor> Eigenize(const OctopusRowVectorXcd & m)
    {
        Eigen::Matrix<std::complex<double>,1,Eigen::Dynamic,Eigen::RowMajor> r(m.cols());
        for(size_t i = 0; i < m.rows(); i++)        
                r(i) = m(i);
        return r;
    }

    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Eigenize(const OctopusMatrixXf & m)
    {
        Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> r(m.rows(),m.cols());
        for(size_t i = 0; i < m.rows(); i++)
            for(size_t j = 0; j < m.cols(); j++)
                r(i,j) = m(i,j);
        return r;
    }
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Eigenize(const OctopusMatrixXd & m)
    {
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> r(m.rows(),m.cols());
        for(size_t i = 0; i < m.rows(); i++)
            for(size_t j = 0; j < m.cols(); j++)
                r(i,j) = m(i,j);
        return r;
    }
    Eigen::Matrix<std::complex<float>,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Eigenize(const OctopusMatrixXcf & m)
    {
        Eigen::Matrix<std::complex<float>,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> r(m.rows(),m.cols());
        for(size_t i = 0; i < m.rows(); i++)
            for(size_t j = 0; j < m.cols(); j++)
                r(i,j) = m(i,j);
        return r;
    }
    Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Eigenize(const OctopusMatrixXcd & m)
    {
        Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> r(m.rows(),m.cols());
        for(size_t i = 0; i < m.rows(); i++)
            for(size_t j = 0; j < m.cols(); j++)
                r(i,j) = m(i,j);
        return r;
    }

    OctopusRowVectorXf Octavate(const Eigen::Matrix<float,1,Eigen::Dynamic,Eigen::RowMajor> & m)
    {
        OctopusRowVectorXf r(m.rows());
        for(size_t i = 0; i < m.rows(); i++)        
                r(i) = m(i);
        return r;
    }
    OctopusRowVectorXd Octavate(const Eigen::Matrix<double,1,Eigen::Dynamic,Eigen::RowMajor> & m)
    {
        OctopusRowVectorXd r(m.rows());
        for(size_t i = 0; i < m.rows(); i++)        
                r(i) = m(i);
        return r;
    }
    OctopusRowVectorXcf Octavate(const Eigen::Matrix<std::complex<float>,1,Eigen::Dynamic,Eigen::RowMajor> & m)
    {
        OctopusRowVectorXcf r(m.rows());
        for(size_t i = 0; i < m.rows(); i++)        
                r(i) = m(i);
        return r;
    }
    OctopusRowVectorXcd Octavate(const Eigen::Matrix<std::complex<double>,1,Eigen::Dynamic,Eigen::RowMajor> & m)
    {
        OctopusRowVectorXcd r(m.rows());
        for(size_t i = 0; i < m.rows(); i++)        
                r(i) = m(i);
        return r;
    }  

    OctopusMatrixXf Octavate(const Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & m)
    {
        OctopusMatrixXf r(m.rows(),m.cols());
        for(size_t i = 0; i < m.rows(); i++)
            for(size_t j = 0; j < m.cols(); j++)
                r(i,j) = m(i,j);
        return r;
    }
    OctopusMatrixXd Octavate(const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & m)
    {
        OctopusMatrixXd r(m.rows(),m.cols());
        for(size_t i = 0; i < m.rows(); i++)
            for(size_t j = 0; j < m.cols(); j++)
                r(i,j) = m(i,j);
        return r;
    }
    OctopusMatrixXcf Octavate(const Eigen::Matrix<std::complex<float>,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & m)
    {
        OctopusMatrixXcf r(m.rows(),m.cols());
        for(size_t i = 0; i < m.rows(); i++)
            for(size_t j = 0; j < m.cols(); j++)
                r(i,j) = m(i,j);
        return r;
    }
    OctopusMatrixXcd Octavate(const Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & m)
    {
        OctopusMatrixXcd r(m.rows(),m.cols());
        for(size_t i = 0; i < m.rows(); i++)
            for(size_t j = 0; j < m.cols(); j++)
                r(i,j) = m(i,j);
        return r;
    }

    /*
    OctopusRowVectorXf Octavate(const Casino::sample_vector<float> & m)
    {
        OctopusRowVectorXf r(m.size());
        for(size_t i = 0; i < m.size(); i++)        
                r(i) = m(i);
        return r;
    }
    OctopusRowVectorXd Octavate(const Casino::sample_vector<double> & m)
    {
        OctopusRowVectorXd r(m.size());
        for(size_t i = 0; i < m.size(); i++)        
                r(i) = m(i);
        return r;
    }
    OctopusRowVectorXcf Octavate(const Casino::complex_vector<float> & m)
    {
        OctopusRowVectorXcf r(m.size());
        for(size_t i = 0; i < m.size(); i++)        
                r(i) = m(i);
        return r;
    }
    OctopusRowVectorXcd Octavate(const Casino::complex_vector<double> & m)
    {
        OctopusRowVectorXcd r(m.size());
        for(size_t i = 0; i < m.size(); i++)        
                r(i) = m(i);
        return r;
    }  

    OctopusColVectorXf Octavate(const Casino::sample_vector<float> & m,bool x)
    {
        OctopusColVectorXf r(m.size());
        for(size_t i = 0; i < m.size(); i++)        
                r(i) = m(i);
        return r;
    }
    OctopusColVectorXd Octavate(const Casino::sample_vector<double> & m,bool x)
    {
        OctopusColVectorXd r(m.size());
        for(size_t i = 0; i < m.size(); i++)        
                r(i) = m(i);
        return r;
    }
    OctopusColVectorXcf Octavate(const Casino::complex_vector<float> & m,bool x)
    {
        OctopusColVectorXcf r(m.size());
        for(size_t i = 0; i < m.size(); i++)        
                r(i) = m(i);
        return r;
    }
    OctopusColVectorXcd Octavate(const Casino::complex_vector<double> & m,bool x)
    {
        OctopusColVectorXcd r(m.size());
        for(size_t i = 0; i < m.size(); i++)        
                r(i) = m(i);
        return r;
    }  

    OctopusMatrixXf Octavate(const Casino::sample_matrix<float> & m)
    {
        OctopusMatrixXf r(m.rows(),m.cols());
        for(size_t i = 0; i < m.rows(); i++)
            for(size_t j = 0; j < m.cols(); j++)
                r(i,j) = m(i,j);
        return r;
    }
    OctopusMatrixXd Octavate(const Casino::sample_matrix<double> & m)
    {
        OctopusMatrixXd r(m.rows(),m.cols());
        for(size_t i = 0; i < m.rows(); i++)
            for(size_t j = 0; j < m.cols(); j++)
                r(i,j) = m(i,j);
        return r;
    }
    OctopusMatrixXcf Octavate(const Casino::complex_matrix<float> & m)
    {
        OctopusMatrixXcf r(m.rows(),m.cols());
        for(size_t i = 0; i < m.rows(); i++)
            for(size_t j = 0; j < m.cols(); j++)
                r(i,j) = m(i,j);
        return r;
    }
    OctopusMatrixXcd Octavate(const Casino::complex_matrix<double> & m)
    {
        OctopusMatrixXcd r(m.rows(),m.cols());
        for(size_t i = 0; i < m.rows(); i++)
            for(size_t j = 0; j < m.cols(); j++)
                r(i,j) = m(i,j);
        return r;
    }           
    */
}