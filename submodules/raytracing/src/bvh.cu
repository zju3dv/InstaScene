// #include <Eigen/Dense>
#include <raytracing/common.h>
#include <raytracing/triangle.cuh>
#include <raytracing/bvh.cuh>

// #ifdef NGP_OPTIX
// #  include <optix.h>
// #  include <optix_stubs.h>
// #  include <optix_function_table_definition.h>
// #  include <optix_stack_size.h>

// // Custom optix toolchain stuff
// #  include "optix/pathescape.h"
// #  include "optix/raystab.h"
// #  include "optix/raytrace.h"

// #  include "optix/program.h"

// // Compiled optix program PTX generated by cmake and wrapped in a C
// // header by bin2c.
// namespace optix_ptx {
// 	#include <optix_ptx.h>
// }
// #endif //NGP_OPTIX

#include <stack>
#include <iostream>
#include <cstdio>

using namespace Eigen;
using namespace raytracing;


namespace raytracing {

constexpr float MAX_DIST = 20.0f;
constexpr float MAX_DIST_SQ = MAX_DIST*MAX_DIST;


// #ifdef NGP_OPTIX
// OptixDeviceContext g_optix;

// namespace optix {
// 	bool initialize() {
// 		static bool ran_before = false;
// 		static bool is_optix_initialized = false;
// 		if (ran_before) {
// 			return is_optix_initialized;
// 		}

// 		ran_before = true;

// 		// Initialize CUDA with a no-op call to the the CUDA runtime API
// 		CUDA_CHECK_THROW(cudaFree(nullptr));

// 		try {
// 			// Initialize the OptiX API, loading all API entry points
// 			OPTIX_CHECK_THROW(optixInit());

// 			// Specify options for this context. We will use the default options.
// 			OptixDeviceContextOptions options = {};

// 			// Associate a CUDA context (and therefore a specific GPU) with this
// 			// device context
// 			CUcontext cuCtx = 0; // NULL means take the current active context

// 			OPTIX_CHECK_THROW(optixDeviceContextCreate(cuCtx, &options, &g_optix));
// 		} catch (std::exception& e) {
// 			tlog::warning() << "OptiX failed to initialize: " << e.what();
// 			return false;
// 		}

// 		is_optix_initialized = true;
// 		return true;
// 	}

// 	class Gas {
// 	public:
// 		Gas(const GPUMemory<Triangle>& triangles, OptixDeviceContext optix, cudaStream_t stream) {
// 			// Specify options for the build. We use default options for simplicity.
// 			OptixAccelBuildOptions accel_options = {};
// 			accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
// 			accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

// 			// Populate the build input struct with our triangle data as well as
// 			// information about the sizes and types of our data
// 			const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
// 			OptixBuildInput triangle_input = {};

// 			CUdeviceptr d_triangles = (CUdeviceptr)(uintptr_t)triangles.data();

// 			triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
// 			triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
// 			triangle_input.triangleArray.numVertices = (uint32_t)triangles.size()*3;
// 			triangle_input.triangleArray.vertexBuffers = &d_triangles;
// 			triangle_input.triangleArray.flags = triangle_input_flags;
// 			triangle_input.triangleArray.numSbtRecords = 1;

// 			// Query OptiX for the memory requirements for our GAS
// 			OptixAccelBufferSizes gas_buffer_sizes;
// 			OPTIX_CHECK_THROW(optixAccelComputeMemoryUsage(optix, &accel_options, &triangle_input, 1, &gas_buffer_sizes));

// 			// Allocate device memory for the scratch space buffer as well
// 			// as the GAS itself
// 			GPUMemory<char> gas_tmp_buffer{gas_buffer_sizes.tempSizeInBytes};
// 			m_gas_gpu_buffer.resize(gas_buffer_sizes.outputSizeInBytes);

// 			OPTIX_CHECK_THROW(optixAccelBuild(
// 				optix,
// 				stream,
// 				&accel_options,
// 				&triangle_input,
// 				1,           // num build inputs
// 				(CUdeviceptr)(uintptr_t)gas_tmp_buffer.data(),
// 				gas_buffer_sizes.tempSizeInBytes,
// 				(CUdeviceptr)(uintptr_t)m_gas_gpu_buffer.data(),
// 				gas_buffer_sizes.outputSizeInBytes,
// 				&m_gas_handle, // Output handle to the struct
// 				nullptr,       // emitted property list
// 				0              // num emitted properties
// 			));
// 		}

// 		OptixTraversableHandle handle() const {
// 			return m_gas_handle;
// 		}

// 	private:
// 		OptixTraversableHandle m_gas_handle;
// 		GPUMemory<char> m_gas_gpu_buffer;
// 	};
// }
// #endif //NGP_OPTIX

// __global__ void signed_distance_watertight_kernel(uint32_t n_elements, const Vector3f* __restrict__ positions, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, float* __restrict__ distances, bool use_existing_distances_as_upper_bounds = false);
// __global__ void signed_distance_raystab_kernel(uint32_t n_elements, const Vector3f* __restrict__ positions, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, float* __restrict__ distances, bool use_existing_distances_as_upper_bounds = false);
// __global__ void unsigned_distance_kernel(uint32_t n_elements, const Vector3f* __restrict__ positions, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, float* __restrict__ distances, bool use_existing_distances_as_upper_bounds = false);
__global__ void raytrace_kernel(uint32_t n_elements, const Vector3f* __restrict__ rays_o, const Vector3f* __restrict__ rays_d, Vector3f* __restrict__ positions, Vector3f* __restrict__ normals,int64_t* __restrict__ face_id, float* __restrict__ depth, const TriangleBvhNode* __restrict__ nodes, const Triangle* __restrict__ triangles);

struct DistAndIdx {
    float dist;
    uint32_t idx;

    // Sort in descending order!
    __host__ __device__ bool operator<(const DistAndIdx& other) {
        return dist < other.dist;
    }
};

template <typename T>
__host__ __device__ void inline compare_and_swap(T& t1, T& t2) {
    if (t1 < t2) {
        T tmp{t1}; t1 = t2; t2 = tmp;
    }
}

// Sorting networks from http://users.telenet.be/bertdobbelaere/SorterHunter/sorting_networks.html#N4L5D3
template <uint32_t N, typename T>
__host__ __device__ void sorting_network(T values[N]) {
    static_assert(N <= 8, "Sorting networks are only implemented up to N==8");
    if (N <= 1) {
        return;
    } else if (N == 2) {
        compare_and_swap(values[0], values[1]);
    } else if (N == 3) {
        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[1], values[2]);
    } else if (N == 4) {
        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[1], values[3]);
        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[1], values[2]);
    } else if (N == 5) {
        compare_and_swap(values[0], values[3]);
        compare_and_swap(values[1], values[4]);

        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[1], values[3]);

        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[4]);

        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[3], values[4]);

        compare_and_swap(values[2], values[3]);
    } else if (N == 6) {
        compare_and_swap(values[0], values[5]);
        compare_and_swap(values[1], values[3]);
        compare_and_swap(values[2], values[4]);

        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[3], values[4]);

        compare_and_swap(values[0], values[3]);
        compare_and_swap(values[2], values[5]);

        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[4], values[5]);

        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[3], values[4]);
    } else if (N == 7) {
        compare_and_swap(values[0], values[6]);
        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[4], values[5]);

        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[1], values[4]);
        compare_and_swap(values[3], values[6]);

        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[5]);
        compare_and_swap(values[3], values[4]);

        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[4], values[6]);

        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[4], values[5]);

        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[3], values[4]);
        compare_and_swap(values[5], values[6]);
    } else if (N == 8) {
        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[1], values[3]);
        compare_and_swap(values[4], values[6]);
        compare_and_swap(values[5], values[7]);

        compare_and_swap(values[0], values[4]);
        compare_and_swap(values[1], values[5]);
        compare_and_swap(values[2], values[6]);
        compare_and_swap(values[3], values[7]);

        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[4], values[5]);
        compare_and_swap(values[6], values[7]);

        compare_and_swap(values[2], values[4]);
        compare_and_swap(values[3], values[5]);

        compare_and_swap(values[1], values[4]);
        compare_and_swap(values[3], values[6]);

        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[3], values[4]);
        compare_and_swap(values[5], values[6]);
    }
}

template <uint32_t BRANCHING_FACTOR>
class TriangleBvhWithBranchingFactor : public TriangleBvh {
public:
    __host__ __device__ static std::pair<int, float> ray_intersect(Ref<const Vector3f> ro, Ref<const Vector3f> rd, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles) {
        FixedIntStack query_stack;
        query_stack.push(0);

        float mint = MAX_DIST;
        int shortest_idx = -1;

        while (!query_stack.empty()) {
            int idx = query_stack.pop();

            const TriangleBvhNode& node = bvhnodes[idx];

            if (node.left_idx < 0) {
                int end = -node.right_idx-1;
                for (int i = -node.left_idx-1; i < end; ++i) {
                    float t = triangles[i].ray_intersect(ro, rd);
                    if (t < mint) {
                        mint = t;
                        shortest_idx = i;
                    }
                }
            } else {
                DistAndIdx children[BRANCHING_FACTOR];

                uint32_t first_child = node.left_idx;

                #pragma unroll
                for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
                    children[i] = {bvhnodes[i+first_child].bb.ray_intersect(ro, rd).x(), i+first_child};
                }

                sorting_network<BRANCHING_FACTOR>(children);

                #pragma unroll
                for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
                    if (children[i].dist < mint) {
                        query_stack.push(children[i].idx);
                    }
                }
            }
        }

        return {shortest_idx, mint};
    }

    // __host__ __device__ static std::pair<int, float> closest_triangle(const Vector3f& point, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, float max_distance_sq = MAX_DIST_SQ) {
    //     FixedIntStack query_stack;
    //     query_stack.push(0);

    //     float shortest_distance_sq = max_distance_sq;
    //     int shortest_idx = -1;

    //     while (!query_stack.empty()) {
    //         int idx = query_stack.pop();

    //         const TriangleBvhNode& node = bvhnodes[idx];

    //         if (node.left_idx < 0) {
    //             int end = -node.right_idx-1;
    //             for (int i = -node.left_idx-1; i < end; ++i) {
    //                 float dist_sq = triangles[i].distance_sq(point);
    //                 if (dist_sq <= shortest_distance_sq) {
    //                     shortest_distance_sq = dist_sq;
    //                     shortest_idx = i;
    //                 }
    //             }
    //         } else {
    //             DistAndIdx children[BRANCHING_FACTOR];

    //             uint32_t first_child = node.left_idx;

    //             #pragma unroll
    //             for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
    //                 children[i] = {bvhnodes[i+first_child].bb.distance_sq(point), i+first_child};
    //             }

    //             sorting_network<BRANCHING_FACTOR>(children);

    //             #pragma unroll
    //             for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
    //                 if (children[i].dist <= shortest_distance_sq) {
    //                     query_stack.push(children[i].idx);
    //                 }
    //             }
    //         }
    //     }

    //     if (shortest_idx == -1) {
    //         // printf("No closest triangle found. This must be a bug! %d\n", BRANCHING_FACTOR);
    //         shortest_idx = 0;
    //         shortest_distance_sq = 0.0f;
    //     }

    //     return {shortest_idx, std::sqrt(shortest_distance_sq)};
    // }

    // // Assumes that "point" is a location on a triangle
    // __host__ __device__ static Vector3f avg_normal_around_point(const Vector3f& point, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles) {
    //     FixedIntStack query_stack;
    //     query_stack.push(0);

    //     static constexpr float EPSILON = 1e-6f;

    //     float total_weight = 0;
    //     Vector3f result = Vector3f::Zero();

    //     while (!query_stack.empty()) {
    //         int idx = query_stack.pop();

    //         const TriangleBvhNode& node = bvhnodes[idx];

    //         if (node.left_idx < 0) {
    //             int end = -node.right_idx-1;
    //             for (int i = -node.left_idx-1; i < end; ++i) {
    //                 if (triangles[i].distance_sq(point) < EPSILON) {
    //                     float weight = 1; // TODO: cot weight
    //                     result += triangles[i].normal();
    //                     total_weight += weight;
    //                 }
    //             }
    //         } else {
    //             uint32_t first_child = node.left_idx;

    //             #pragma unroll
    //             for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
    //                 if (bvhnodes[i+first_child].bb.distance_sq(point) < EPSILON) {
    //                     query_stack.push(i+first_child);
    //                 }
    //             }
    //         }
    //     }

    //     return result / total_weight;
    // }

    // __host__ __device__ static float unsigned_distance(const Vector3f& point, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, float max_distance_sq = MAX_DIST_SQ) {
    //     return closest_triangle(point, bvhnodes, triangles, max_distance_sq).second;
    // }

    // __host__ __device__ static float signed_distance_watertight(const Vector3f& point, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, float max_distance_sq = MAX_DIST_SQ) {
    //     auto p = closest_triangle(point, bvhnodes, triangles, max_distance_sq);

    //     const Triangle& tri = triangles[p.first];
    //     Vector3f closest_point = tri.closest_point(point);
    //     Vector3f avg_normal = avg_normal_around_point(closest_point, bvhnodes, triangles);

    //     return std::copysignf(p.second, avg_normal.dot(point - closest_point));
    // }

    // __host__ __device__ static float signed_distance_raystab(const Vector3f& point, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, float max_distance_sq = MAX_DIST_SQ, default_rng_t rng={}) {
    //     float distance = unsigned_distance(point, bvhnodes, triangles, max_distance_sq);

    //     Vector2f offset = random_val_2d(rng);

    //     static constexpr uint32_t N_STAB_RAYS = 32;
    //     for (uint32_t i = 0; i < N_STAB_RAYS; ++i) {
    //         // Use a Fibonacci lattice (with random offset) to regularly
    //         // distribute the stab rays over the sphere.
    //         Vector3f d = fibonacci_dir<N_STAB_RAYS>(i, offset);

    //         // If any of the stab rays goes outside the mesh, the SDF is positive.
    //         if (ray_intersect(point, -d, bvhnodes, triangles).first < 0 || ray_intersect(point, d, bvhnodes, triangles).first < 0) {
    //             return distance;
    //         }
    //     }

    //     return -distance;
    // }

    // // Assumes that "point" is a location on a triangle
    // Vector3f avg_normal_around_point(const Vector3f& point, const Triangle* __restrict__ triangles) const {
    //     return avg_normal_around_point(point, m_nodes.data(), triangles);
    // }

    // float signed_distance(EMeshSdfMode mode, const Vector3f& point, const std::vector<Triangle>& triangles) const {
    //     if (mode == EMeshSdfMode::Watertight) {
    //         return signed_distance_watertight(point, m_nodes.data(), triangles.data());
    //     } else {
    //         return signed_distance_raystab(point, m_nodes.data(), triangles.data());
    //     }
    // }

    // void signed_distance_gpu(uint32_t n_elements, EMeshSdfMode mode, const Vector3f* gpu_positions, float* gpu_distances, const Triangle* gpu_triangles, bool use_existing_distances_as_upper_bounds, cudaStream_t stream) override {
    //     if (mode == EMeshSdfMode::Watertight) {
    //         linear_kernel(signed_distance_watertight_kernel, 0, stream,
    //             n_elements,
    //             gpu_positions,
    //             m_nodes_gpu.data(),
    //             gpu_triangles,
    //             gpu_distances,
    //             use_existing_distances_as_upper_bounds
    //         );
    //     } else {
    //         {
    //             if (mode == EMeshSdfMode::Raystab) {
    //                 linear_kernel(signed_distance_raystab_kernel, 0, stream,
    //                     n_elements,
    //                     gpu_positions,
    //                     m_nodes_gpu.data(),
    //                     gpu_triangles,
    //                     gpu_distances,
    //                     use_existing_distances_as_upper_bounds
    //                 );
    //             } else if (mode == EMeshSdfMode::PathEscape) {
    //                 throw std::runtime_error{"TriangleBvh: EMeshSdfMode::PathEscape is only supported with OptiX enabled."};
    //             }
    //         }
    //     }
    // }

    void ray_trace_gpu(uint32_t n_elements, const float* rays_o, const float* rays_d, float* positions, float* normals, int64_t* face_id, float* depth, const Triangle* gpu_triangles, cudaStream_t stream) override {

        // cast float* to Vector3f*
        const Vector3f* rays_o_vec = (const Vector3f*)rays_o;
        const Vector3f* rays_d_vec = (const Vector3f*)rays_d;
        Vector3f* positions_vec = (Vector3f*)positions;
        Vector3f* normals_vec = (Vector3f*)normals;

// #ifdef NGP_OPTIX
//         if (m_optix.available) {
//             m_optix.raytrace->invoke({rays_o_vec, rays_d_vec, gpu_triangles, m_optix.gas->handle()}, {n_elements, 1, 1}, stream);
//         } else
// #endif //NGP_OPTIX
        {
            linear_kernel(raytrace_kernel, 0, stream,
                n_elements,
                rays_o_vec,
                rays_d_vec,
                positions_vec,
                normals_vec,
                face_id,
                depth,
                m_nodes_gpu.data(),
                gpu_triangles
            );
        }
    }

    // bool touches_triangle(const BoundingBox& bb, const TriangleBvhNode& node, const Triangle* __restrict__ triangles) const {
    //     if (!node.bb.intersects(bb)) {
    //         return false;
    //     }

    //     if (node.left_idx < 0) {
    //         // Touches triangle leaves?
    //         int end = -node.right_idx-1;
    //         for (int i = -node.left_idx-1; i < end; ++i) {
    //             if (bb.intersects(triangles[i])) {
    //                 return true;
    //             }
    //         }
    //     } else {
    //         // Touches children?
    //         int child_idx = node.left_idx;
    //         for (int i = 0; i < BRANCHING_FACTOR; ++i) {
    //             if (touches_triangle(bb, m_nodes[i+child_idx], triangles)) {
    //                 return true;
    //             }
    //         }
    //     }

    //     return false;
    // }

    // bool touches_triangle(const BoundingBox& bb, const Triangle* __restrict__ triangles) const override {
    //     return touches_triangle(bb, m_nodes.front(), triangles);
    // }

    void build(std::vector<Triangle>& triangles, uint32_t n_primitives_per_leaf) override {
        m_nodes.clear();

        // Root
        m_nodes.emplace_back();
        m_nodes.front().bb = BoundingBox(std::begin(triangles), std::end(triangles));

        struct BuildNode {
            int node_idx;
            std::vector<Triangle>::iterator begin;
            std::vector<Triangle>::iterator end;
        };

        std::stack<BuildNode> build_stack;
        build_stack.push({0, std::begin(triangles), std::end(triangles)});

        while (!build_stack.empty()) {
            const BuildNode& curr = build_stack.top();
            size_t node_idx = curr.node_idx;

            std::array<BuildNode, BRANCHING_FACTOR> children;
            children[0].begin = curr.begin;
            children[0].end = curr.end;

            build_stack.pop();

            // Partition the triangles into the children
            int n_children = 1;
            while (n_children < BRANCHING_FACTOR) {
                for (int i = n_children - 1; i >= 0; --i) {
                    auto& child = children[i];

                    // Choose axis with maximum standard deviation
                    Vector3f mean = Vector3f::Zero();
                    for (auto it = child.begin; it != child.end; ++it) {
                        mean += it->centroid();
                    }
                    mean /= (float)std::distance(child.begin, child.end);

                    Vector3f var = Vector3f::Zero();
                    for (auto it = child.begin; it != child.end; ++it) {
                        Vector3f diff = it->centroid() - mean;
                        var += diff.cwiseProduct(diff);
                    }
                    var /= (float)std::distance(child.begin, child.end);

                    Vector3f::Index axis;
                    var.maxCoeff(&axis);

                    auto m = child.begin + std::distance(child.begin, child.end)/2;
                    std::nth_element(child.begin, m, child.end, [&](const Triangle& tri1, const Triangle& tri2) { return tri1.centroid(axis) < tri2.centroid(axis); });

                    children[i*2].begin = children[i].begin;
                    children[i*2+1].end = children[i].end;
                    children[i*2].end = children[i*2+1].begin = m;
                }

                n_children *= 2;
            }

            // Create next build nodes
            m_nodes[node_idx].left_idx = (int)m_nodes.size();
            for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
                auto& child = children[i];
                assert(child.begin != child.end);
                child.node_idx = (int)m_nodes.size();

                m_nodes.emplace_back();
                m_nodes.back().bb = BoundingBox(child.begin, child.end);

                if (std::distance(child.begin, child.end) <= n_primitives_per_leaf) {
                    m_nodes.back().left_idx = -(int)std::distance(std::begin(triangles), child.begin)-1;
                    m_nodes.back().right_idx = -(int)std::distance(std::begin(triangles), child.end)-1;
                } else {
                    build_stack.push(child);
                }
            }
            m_nodes[node_idx].right_idx = (int)m_nodes.size();
        }

        m_nodes_gpu.resize_and_copy_from_host(m_nodes);

        // std::cout << "[INFO] Built TriangleBvh: nodes=" << m_nodes.size() << std::endl;
    }

//     void build_optix(const GPUMemory<Triangle>& triangles, cudaStream_t stream) override {
// #ifdef NGP_OPTIX
//         m_optix.available = optix::initialize();
//         if (m_optix.available) {
//             m_optix.gas = std::make_unique<optix::Gas>(triangles, g_optix, stream);
//             m_optix.raystab = std::make_unique<optix::Program<Raystab>>((const char*)optix_ptx::raystab_ptx, sizeof(optix_ptx::raystab_ptx), g_optix);
//             m_optix.raytrace = std::make_unique<optix::Program<Raytrace>>((const char*)optix_ptx::raytrace_ptx, sizeof(optix_ptx::raytrace_ptx), g_optix);
//             m_optix.pathescape = std::make_unique<optix::Program<PathEscape>>((const char*)optix_ptx::pathescape_ptx, sizeof(optix_ptx::pathescape_ptx), g_optix);
//             tlog::success() << "Built OptiX GAS and shaders";
//         } else {
//             tlog::warning() << "Falling back to slower TriangleBVH::ray_intersect.";
//         }
// #else //NGP_OPTIX
//         tlog::warning() << "OptiX was not built. Falling back to slower TriangleBVH::ray_intersect.";
// #endif //NGP_OPTIX
//     }

    TriangleBvhWithBranchingFactor() {}

// private:
// #ifdef NGP_OPTIX
//     struct {
//         std::unique_ptr<optix::Gas> gas;
//         std::unique_ptr<optix::Program<Raystab>> raystab;
//         std::unique_ptr<optix::Program<Raytrace>> raytrace;
//         std::unique_ptr<optix::Program<PathEscape>> pathescape;
//         bool available = false;
//     } m_optix;
// #endif //NGP_OPTIX
};

using TriangleBvh4 = TriangleBvhWithBranchingFactor<4>;

std::unique_ptr<TriangleBvh> TriangleBvh::make() {
    return std::unique_ptr<TriangleBvh>(new TriangleBvh4());
}

// __global__ void signed_distance_watertight_kernel(uint32_t n_elements,
//     const Vector3f* __restrict__ positions,
//     const TriangleBvhNode* __restrict__ bvhnodes,
//     const Triangle* __restrict__ triangles,
//     float* __restrict__ distances,
//     bool use_existing_distances_as_upper_bounds
// ) {
//     uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= n_elements) return;

//     float max_distance = use_existing_distances_as_upper_bounds ? distances[i] : MAX_DIST;
//     distances[i] = TriangleBvh4::signed_distance_watertight(positions[i], bvhnodes, triangles, max_distance*max_distance);
// }

// __global__ void signed_distance_raystab_kernel(
//     uint32_t n_elements,
//     const Vector3f* __restrict__ positions,
//     const TriangleBvhNode* __restrict__ bvhnodes,
//     const Triangle* __restrict__ triangles,
//     float* __restrict__ distances,
//     bool use_existing_distances_as_upper_bounds
// ) {
//     uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= n_elements) return;

//     float max_distance = use_existing_distances_as_upper_bounds ? distances[i] : MAX_DIST;
//     default_rng_t rng;
//     rng.advance(i * 2);

//     distances[i] = TriangleBvh4::signed_distance_raystab(positions[i], bvhnodes, triangles, max_distance*max_distance, rng);
// }

// __global__ void unsigned_distance_kernel(uint32_t n_elements,
//     const Vector3f* __restrict__ positions,
//     const TriangleBvhNode* __restrict__ bvhnodes,
//     const Triangle* __restrict__ triangles,
//     float* __restrict__ distances,
//     bool use_existing_distances_as_upper_bounds
// ) {
//     uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= n_elements) return;

//     float max_distance = use_existing_distances_as_upper_bounds ? distances[i] : MAX_DIST;
//     distances[i] = TriangleBvh4::unsigned_distance(positions[i], bvhnodes, triangles, max_distance*max_distance);
// }

__global__ void raytrace_kernel(uint32_t n_elements, const Vector3f* __restrict__ rays_o, const Vector3f* __restrict__ rays_d, Vector3f* __restrict__ positions, Vector3f* __restrict__ normals,int64_t* __restrict__ face_id, float* __restrict__ depth, const TriangleBvhNode* __restrict__ nodes, const Triangle* __restrict__ triangles) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    Vector3f ro = rays_o[i];
    Vector3f rd = rays_d[i];

    auto p = TriangleBvh4::ray_intersect(ro, rd, nodes, triangles);

    // write depth
    depth[i] = p.second;
 
    // intersection point is written back to positions.
    // non-intersect point reaches at most 10 depth
    positions[i] = ro + p.second * rd;

    // face normal is written to directions.
    if (p.first >= 0) {
        normals[i] = triangles[p.first].normal();
        face_id[i] = triangles[p.first].id;
    } else {
        normals[i].setZero();
        face_id[i] = -1;
    }

    // shall we write the depth? (p.second)

}
    
}