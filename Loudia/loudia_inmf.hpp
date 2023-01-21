#pragma once

namespace Loudia
{
    class INMF {
    protected:
        // Internal parameters
        int _fftSize;
        int _numComponents;

        int _maxIterations;

        Real _eps;

        int _numPast;

        // Coefficients between 0 and 1 which represent
        // how much should the past and the new be taken
        // into account
        Real _pastCoeff;
        Real _newCoeff;
        
        // Internal variables
        MatrixXR _H, _V, _W, _VH, _HH;

    public:
        INMF(int fftSize, int numComponents, int numPast, Real pastCoeff,  int maxIterations = 10, Real eps = 1e-9);

        ~INMF();

        void setup();

        void process(const MatrixXR& v, MatrixXR* w, MatrixXR* h);

        void reset();

    };

    INMF::INMF(int fftSize, int numComponents, int numPast, Real pastCoeff, int maxIterations, Real eps) :
    _fftSize( fftSize ),
    _numComponents( numComponents ),
    _maxIterations( maxIterations ),
    _eps( eps ),
    _numPast( numPast ),
    _pastCoeff( pastCoeff ),
    _newCoeff( 1 - pastCoeff )
    {
    
    LOUDIA_DEBUG("INMF: Constructor fftSize: " << _fftSize
            << " numComponents: " << _numComponents
            << " numPast: " << _numPast
            << " pastCoeff: " << _pastCoeff
            << " maxIterations: " << _maxIterations );
    
    setup();
    }

    INMF::~INMF() {}


    void INMF::setup() {
    // Prepare the buffers
    LOUDIA_DEBUG("INMF: Setting up...");
    
    _V.resize(_numPast, _fftSize);
    _H.resize(_numPast, _numComponents);
    _W.resize(_numComponents, _fftSize);
    
    reset();

    LOUDIA_DEBUG("INMF: Finished set up...");
    }


    void INMF::process(const MatrixXR& v, MatrixXR* w, MatrixXR* h) {
    LOUDIA_DEBUG("INMF: Processing windowed");
    const int rows = v.rows();
    const int cols = v.cols();
    
    if ( cols != _numComponents ) {
        // Throw error wrong number of columns
    }

    // The X matrix is spectrumAbs.transpose()
    // Some beleive it can be useful to normalize  
    // The W matrix is (*w).transpose()
    (*w) = _W;
    
    // The H matrix is (*h).transpose()
    (*h).resize(rows, _numComponents);
    
    // Initializing h
    // TODO: initialize with a Normal distribution
    (*h).setRandom();
    (*h) = (*h).array().abs();

    LOUDIA_DEBUG("INMF: Begin rows");
    for (int row = 0; row < rows; row++ ) {

        // Calculate beta * VHt
        _VH = _pastCoeff * (_V.transpose() * _H);

        // Calculate beta * HHt
        _HH = _pastCoeff * (_H.transpose() * _H);

        LOUDIA_DEBUG("INMF: Begin iterations");
        for ( int iter = 0; iter < _maxIterations; iter++ ) {
        /*
        MatrixXR Wv = (*w) * v.row(row).transpose();
        MatrixXR WWh = ((*w) * (*w).transpose()) * (*h).row(row).transpose();
        for ( int a = 0; a < _numComponents; a++ ) {
            (*h).row( row )(a) *= Wv(a) / WWh(a);
        }
        */

        //DEBUG("INMF: Update h");
        // Eq. 9 in Bucak 2008
        /*
        cout << "---------------" << endl;
        cout << "(*w): " << (*w).rows() << " , " << (*w).cols() << endl;
        cout << "---------------" << endl;
        cout << "(*h): " << (*h).rows() << " , " << (*h).cols() << endl;
        cout << "---------------" << endl;
        cout << "v: " << v.rows() << " , " << v.cols() << endl;
        cout << "---------------" << endl;
        */
        (*h).row( row ).array() *= (((*w) * v.row(row).transpose()).array() / ((((*w) * (*w).transpose()) * (*h).row( row ).transpose()).array() + _eps)).transpose();

        //DEBUG("INMF: Update W");
        // Eq. 13 in Bucak 2008
        /*
        cout << "---------------" << endl;
        cout << "(*w): " << (*w).rows() << " , " << (*w).cols() << endl;
        cout << "_HH: " << _HH.rows() << " , " << _HH.cols() << endl;
        cout << "(*h): " << (*h).rows() << " , " << (*h).cols() << endl;
        cout << "_VH: " << _VH.rows() << " , " << _VH.cols() << endl;
        cout << "---------------" << endl;
        cout << ((*w).transpose() * _HH) << endl;
        cout << "============" << endl;
        cout << (_newCoeff * ((*h).row( row ).transpose() * (*h).row( row ))) << endl;
        cout << "**********" << endl;
        */
        (*w).array() *= ((_VH + (_newCoeff * v.row( row ).transpose() * (*h).row( row ))).array() / (((*w).transpose() * (_HH + _newCoeff * (*h).row( row ).transpose() * (*h).row( row ))).array() + _eps)).transpose();
        }

        LOUDIA_DEBUG("INMF: Shift and update H");
        // Update the past H
        // TODO: when Eigen will have rotate use this
        //_H.rowwise().rotate(-1);
        rowShift(&_H, -1);
        _H.row( _numPast - 1 ) = (*h).row( row );

        LOUDIA_DEBUG("INMF: Shift and update V");
        // Update the past V
        // TODO: when Eigen will have rotate use this
        //_V.rowwise().rotate(-1);
        rowShift(&_V, -1);
        _V.row( _numPast - 1 ) = v.row( row );
        
        LOUDIA_DEBUG("INMF: Keep W");

    }

    // Keep the past W
    _W = (*w);
    
    LOUDIA_DEBUG("INMF: Finished Processing");
    }

    void INMF::reset() {
    // Initial W, H and V
    // TODO: initialize with a Normal distribution
    _W.setRandom();
    _W = _W.array().abs();

    _H.setRandom();
    _H = _H.array().abs();

    _V.setRandom();  
    _V = _V.array().abs();
    }

}