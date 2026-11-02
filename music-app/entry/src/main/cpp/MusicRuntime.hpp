#pragma once
// Compatibility facade; model-independent pipeline and App inventory are separate.
#include "../../../../../runtime/include/levo/MusicPipeline.hpp"
#include "MusicAssets.hpp"
static Json musicCheckAssets(const std::string& root){return LevoMusic::checkAssets(root,musicAssetInventory());}
static Json musicGenerate(const std::string& root,const std::string& lyrics,const std::string& style,int frames,int steps=2,bool fixedLength=false,bool lowLm=false){
    return LevoMusic::generate(root,lyrics,style,frames,musicAssetInventory(),steps,fixedLength,lowLm);
}
