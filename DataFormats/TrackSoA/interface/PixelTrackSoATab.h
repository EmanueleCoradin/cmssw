#ifndef DataFormats_TrackSoA_PixelTrackSoATab_h
#define DataFormats_TrackSoA_PixelTrackSoATab_h

#include <Eigen/Core>
#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include <cmath>

class PixelTrackSoATab {
    // Indices in the state vector
    static constexpr int iPhi = 0;
    static constexpr int iTip = 1;
    static constexpr int iQoverP = 2;
    static constexpr int iCotanTheta = 3;
    static constexpr int iZip = 4;

    // Indices in the covariance matrix
    static constexpr int cPhi = 0;
    static constexpr int cTip = 5;
    static constexpr int cInvPt = 9;
    static constexpr int cZip = 14;

public:
    PixelTrackSoATab() = default;

    PixelTrackSoATab(const reco::TrackSoAConstView& tracks, int idx)
    {
        pt_ = tracks.pt(idx);
        eta_ = tracks.eta(idx);
        phi_ = tracks.state(idx)(iPhi);
        charge_ = (tracks.state(idx)(iQoverP) > 0.f) - (tracks.state(idx)(iQoverP) < 0.f);
        dxy_ = tracks.state(idx)(iTip);
        dz_ = tracks.state(idx)(iZip);
        dxyError_ = std::sqrt(tracks.covariance(idx)(cTip));
        dzError_  = std::sqrt(tracks.covariance(idx)(cZip));
        phiError_ = std::sqrt(tracks.covariance(idx)(cPhi));
        ptError_  = std::sqrt(tracks.covariance(idx)(cInvPt)) * pt_ * pt_;
        chi2_ = tracks.chi2(idx) * (2 * reco::nHits(tracks, idx) - 5);
        nHits_ = reco::nHits(tracks, idx);
        ndof_ = nHits_ * 2 - 5;
    }

    float pt() const { return pt_; }
    float eta() const { return eta_; }
    float phi() const { return phi_; }
    int charge() const { return charge_; }
    float dxy() const { return dxy_; }
    float dz() const { return dz_; }
    float dxyError() const { return dxyError_; }
    float dzError() const { return dzError_; }
    float phiError() const { return phiError_; }
    float ptError() const { return ptError_; }
    float chi2() const { return chi2_; }
    int nHits() const { return nHits_; }
    int ndof() const { return ndof_; }

private:
    float pt_, eta_, phi_;
    int charge_;
    float dxy_, dz_;
    float dxyError_, dzError_, phiError_;
    float ptError_, chi2_;
    int nHits_;
    int ndof_;
};
#endif
