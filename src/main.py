"""
Main application for Ultimate Scene-to-Product Matcher
"""

import asyncio
import os
import time
from pathlib import Path

from core.llm_matcher import build_ultimate_matcher
from utils.logging_config import setup_logging

# Setup logging
logger = setup_logging(log_level="INFO")

async def main():
    """Main application entry point"""
    
    print("🚀 ULTIMATE SCENE-TO-PRODUCT MATCHER")
    print("=" * 60)
    
    # Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    CATALOG_PATH = "data/product-catalog.csv"
    SCENE_IMAGE = "data/example_scene.webp"
    CACHE_DIR = "./cache"
    
    # Verify files exist
    if not Path(CATALOG_PATH).exists():
        print(f"❌ Error: Catalog file not found: {CATALOG_PATH}")
        return
    
    if not Path(SCENE_IMAGE).exists():
        print(f"❌ Error: Scene image not found: {SCENE_IMAGE}")
        return
    
    try:
        print("🔧 Initializing Ultimate Matcher...")
        start_time = time.time()
        
        # Build the ultimate matcher
        matcher = await build_ultimate_matcher(
            openai_api_key=OPENAI_API_KEY,
            catalog_path=CATALOG_PATH,
            cache_dir=CACHE_DIR
        )
        
        init_time = time.time() - start_time
        print(f"✅ Initialization complete in {init_time:.2f}s")
        
        print(f"\n🎯 Getting recommendations for: {SCENE_IMAGE}")
        
        # Get recommendations
        recommendations = await matcher.get_recommendations(
            scene_image_path=SCENE_IMAGE,
            k=5
        )
        
        # Display results
        print("\n🎉 ULTIMATE MATCHER RESULTS")
        print("=" * 60)
        
        # Performance metrics
        metrics = recommendations['performance_metrics']
        print(f"⚡ Response Time: {metrics['total_time_seconds']}s")
        print(f"📊 Average Score: {metrics['avg_score']:.3f}")
        print(f"🎨 Category Diversity: {metrics['category_diversity']} categories")
        print(f"🧠 LLM Re-ranked: {metrics['llm_reranked_count']}/{len(recommendations['recommendations'])} products")
        
        quality_dist = metrics['quality_distribution']
        print(f"💎 Quality Distribution: Premium({quality_dist['premium']}), Standard({quality_dist['standard']}), Basic({quality_dist['basic']})")
        
        # System stats
        stats = recommendations['system_stats']
        print(f"📈 System Performance: {stats['total_products']} products indexed, {stats['total_requests']} total requests")
        print(f"🚀 Average Response Time: {stats['avg_response_time']}s")
        print(f"🤖 LLM Re-ranking: {'Enabled' if stats['llm_reranking_enabled'] else 'Disabled'}")
        
        # Methodology
        methodology = recommendations['methodology']
        print(f"\n🔬 METHODOLOGY:")
        print(f"   Retrieval: {methodology['retrieval']}")
        print(f"   Scene Analysis: {methodology['scene_analysis']}")
        print(f"   Product Enhancement: {methodology['product_enhancement']}")
        print(f"   Ranking: {methodology['ranking']}")
        print(f"   Scoring: {methodology['scoring_weights']}")
        
        print(f"\n🏆 TOP {len(recommendations['recommendations'])} RECOMMENDATIONS:")
        print("-" * 60)
        
        for rec in recommendations['recommendations']:
            print(f"\n{rec['rank']}. {rec['description']}")
            print(f"   🏷️  Category: {rec['category']} | Style: {rec['style']} | Size: {rec['size']}")
            print(f"   💯 Final Score: {rec['final_score']} (Visual: {rec['similarity_score']}, LLM: {rec['llm_score']})")
            print(f"   🎨 Materials: {', '.join(rec['materials'])} | Colors: {', '.join(rec['colors'])}")
            print(f"   📍 Placement: {rec['placement_suggestion']}")
            print(f"   🧠 Reasoning: {rec['reasoning']}")
            print(f"   💎 Quality: {rec['quality']} | Confidence: {rec['confidence']}")
            
            # Show LLM re-ranking insights
            llm_insights = rec['llm_insights']
            if llm_insights['reranked']:
                print(f"   🎯 LLM Re-ranking: Style Harmony({llm_insights['style_harmony']:.3f}), "
                      f"Functional({llm_insights['functional_appropriateness']:.3f}), "
                      f"Visual Impact({llm_insights['visual_impact']:.3f})")
                print(f"   🏅 LLM Recommendation: {llm_insights['recommendation_level'].upper()}")
            else:
                print(f"   🎯 LLM Re-ranking: Not applied")
        
        print(f"\n✨ Search completed successfully!")
        
        # Performance summary
        total_time = init_time + metrics['total_time_seconds']
        reranked_ratio = metrics['llm_reranked_count'] / len(recommendations['recommendations']) if recommendations['recommendations'] else 0
        
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Initialization: {init_time:.2f}s")
        print(f"   Recommendation: {metrics['total_time_seconds']}s")
        print(f"   Products Processed: {stats['total_products']}")
        print(f"   LLM Enhancement Rate: {reranked_ratio:.1%}")
        print(f"   System Efficiency: {stats['total_products']/total_time:.1f} products/second indexed")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

async def benchmark_performance():
    """Benchmark system performance"""
    
    print("🏃 PERFORMANCE BENCHMARK")
    print("=" * 40)
    
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY required for benchmark")
        return
    
    # Build matcher
    matcher = await build_ultimate_matcher(
        openai_api_key=OPENAI_API_KEY,
        catalog_path="data/product-catalog.csv"
    )
    
    # Run multiple queries to measure performance
    test_scenes = ["data/example_scene.webp"] * 5  # Simulate 5 requests
    
    print(f"🔄 Running {len(test_scenes)} test queries...")
    
    start_time = time.time()
    results = []
    
    for i, scene in enumerate(test_scenes):
        query_start = time.time()
        result = await matcher.get_recommendations(scene, k=5)
        query_time = time.time() - query_start
        
        results.append({
            'query': i + 1,
            'time': query_time,
            'avg_score': result['performance_metrics']['avg_score'],
            'categories': result['performance_metrics']['category_diversity']
        })
        
        print(f"   Query {i+1}: {query_time:.3f}s, Score: {result['performance_metrics']['avg_score']:.3f}")
    
    total_time = time.time() - start_time
    avg_time = total_time / len(test_scenes)
    avg_score = sum(r['avg_score'] for r in results) / len(results)
    
    print(f"\n📊 BENCHMARK RESULTS:")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"   Average Query Time: {avg_time:.3f}s")
    print(f"   Average Score: {avg_score:.3f}")
    print(f"   Queries per Second: {len(test_scenes)/total_time:.1f}")
    print(f"   🎯 Target: <1s per query ({'✅ PASSED' if avg_time < 1.0 else '❌ FAILED'})")

def interactive_mode():
    """Interactive CLI mode"""
    
    print("🔄 INTERACTIVE MODE")
    print("Commands: 'recommend <image_path>', 'benchmark', 'quit'")
    
    # This would need to be implemented with proper async handling
    print("Interactive mode would be implemented here...")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "benchmark":
            asyncio.run(benchmark_performance())
        elif command == "interactive":
            interactive_mode()
        else:
            print("Available commands: benchmark, interactive")
            print("Or run without arguments for demo mode")
    else:
        # Default demo mode
        asyncio.run(main())