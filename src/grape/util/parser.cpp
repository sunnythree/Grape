#include <fstream>
#include "grape/util/parser.h"
#include "cereal/types/vector.hpp"

namespace Grape
{
    void Parse(NetParams &params,std::string path)
    {
        std::ifstream is(path);
        cereal::JSONInputArchive archive(is);
        archive(cereal::make_nvp("net",params));
    }

    void Serialize(NetParams &params,std::string path)
    {
        std::ofstream os(path);
        cereal::JSONOutputArchive archive(os);
        archive(cereal::make_nvp("net",params));
    }
} // namespace Grape
