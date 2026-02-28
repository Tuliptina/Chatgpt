#!/usr/bin/env python3
"""
SLM Memory Management CLI

Commands:
  add       - Add memory with category and folder
  list      - List memories by tier or folder
  search    - Search memories
  folders   - Manage folders
  links     - View neural links
  stats     - System statistics
  cleanup   - Run maintenance
"""

import os

import sys
import json
import argparse
from slm_memory import (
    SLMMemoryManager, 
    MemoryCategory, 
    MemoryTier,
    LinkType,
    SLMConfig
)

# Configuration
HF_KEY = os.environ.get("HF_KEY", "")
MNEMO_URL = "https://athelaperk-mnemo-mcp.hf.space"


def create_manager():
    return SLMMemoryManager(mnemo_url=MNEMO_URL, hf_key=HF_KEY)


def cmd_add(args):
    """Add a memory"""
    manager = create_manager()
    
    category = MemoryCategory[args.category.upper()]
    folder = args.folder or f"/{category.value}"
    
    memory = manager.add_memory(
        content=args.content,
        category=category,
        folder=folder,
        importance=args.importance,
        source="manual"
    )
    
    if memory:
        print(f"✅ Memory added: {memory.id}")
        print(f"   Category: {category.value}")
        print(f"   Folder: {folder}")
        print(f"   Content: {args.content[:50]}...")
    else:
        print("❌ Failed to add memory")


def cmd_list(args):
    """List memories"""
    manager = create_manager()
    
    print("=" * 60)
    print("MEMORY LIST")
    print("=" * 60)
    
    # Get from Mnemo
    memories = manager._mnemo_list()
    
    if args.folder:
        # Filter by folder (would need metadata)
        print(f"Folder filter: {args.folder}")
    
    if args.category:
        # Filter by category
        cat_filter = args.category.lower()
        memories = [m for m in memories if cat_filter in m.get("content", "").lower()]
    
    print(f"\nTotal: {len(memories)} memories\n")
    
    for mem in memories[:args.limit]:
        mid = mem.get("id", "N/A")
        content = mem.get("content", "")[:60]
        tier = mem.get("tier", "semantic")
        print(f"[{mid}] ({tier}) {content}...")
    
    if len(memories) > args.limit:
        print(f"\n... and {len(memories) - args.limit} more")


def cmd_search(args):
    """Search memories"""
    manager = create_manager()
    
    print(f"Searching for: {args.query}\n")
    
    results = manager.retrieve(args.query, top_k=args.limit)
    
    print(f"Found {len(results)} results:\n")
    
    for memory, score in results:
        print(f"[{score:.3f}] {memory.content[:70]}...")


def cmd_folders(args):
    """Manage folders"""
    manager = create_manager()
    
    if args.action == "list":
        print("📁 FOLDERS")
        print("-" * 40)
        for folder in manager.list_folders():
            print(f"  {folder.path}")
            if folder.description:
                print(f"    └─ {folder.description}")
    
    elif args.action == "create":
        if not args.name:
            print("❌ Need --name for folder")
            return
        
        folder = manager.create_folder(
            name=args.name,
            parent=args.parent or "/",
            description=args.description or ""
        )
        print(f"✅ Created folder: {folder.path}")
    
    elif args.action == "show":
        if not args.path:
            print("❌ Need --path to show folder")
            return
        
        memories = manager.get_memories_in_folder(args.path)
        print(f"📁 {args.path}: {len(memories)} memories")
        for mem in memories:
            print(f"  • {mem.content[:50]}...")


def cmd_links(args):
    """View neural links"""
    manager = create_manager()
    
    stats = manager.link_manager.get_stats()
    
    print("🔗 NEURAL LINKS")
    print("-" * 40)
    print(f"Total links: {stats['total_links']}")
    print(f"Average strength: {stats['avg_strength']:.3f}")
    print("\nBy type:")
    for link_type, count in stats['by_type'].items():
        if count > 0:
            print(f"  {link_type}: {count}")


def cmd_stats(args):
    """System statistics"""
    manager = create_manager()
    
    stats = manager.get_stats()
    
    print("📊 SLM MEMORY SYSTEM STATS")
    print("=" * 60)
    
    print("\n🗃️ MEMORY TIERS")
    print(f"  Working Memory: {stats['tiers']['working']} / {SLMConfig.WORKING_MEMORY_MAX_ITEMS}")
    print(f"  Token Memory:   {stats['tiers']['token']}")
    print(f"  Semantic (Mnemo): {stats['tiers']['semantic']}")
    
    print("\n📁 FOLDERS")
    print(f"  Total: {stats['folders']['count']}")
    
    print("\n🔗 NEURAL LINKS")
    print(f"  Total: {stats['links']['total_links']}")
    print(f"  Avg strength: {stats['links']['avg_strength']:.3f}")
    
    print("\n⚙️ CONFIGURATION")
    print(f"  Promotion threshold: {stats['config']['promotion_threshold']}")
    print(f"  Demotion threshold: {stats['config']['demotion_threshold']}")


def cmd_cleanup(args):
    """Run maintenance"""
    manager = create_manager()
    
    print("🧹 Running maintenance...")
    
    result = manager.run_maintenance()
    
    print(f"  Working memory after decay: {result['working_memory_after_decay']}")
    print(f"  Items in token memory: {result['demoted_to_token']}")
    print(f"  Links eligible for pruning: {result['links_prunable']}")
    
    print("✅ Maintenance complete")


def cmd_delete(args):
    """Delete a memory"""
    manager = create_manager()
    
    if manager._mnemo_delete(args.memory_id):
        print(f"✅ Deleted: {args.memory_id}")
    else:
        print(f"❌ Failed to delete: {args.memory_id}")


def main():
    parser = argparse.ArgumentParser(
        description="SLM Memory Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Add command
    add_parser = subparsers.add_parser("add", help="Add a memory")
    add_parser.add_argument("content", help="Memory content")
    add_parser.add_argument("-c", "--category", default="general",
                           choices=["character", "plot", "setting", "theme", "style", "fact", "preference", "general"])
    add_parser.add_argument("-f", "--folder", help="Folder path")
    add_parser.add_argument("-i", "--importance", type=float, default=0.5, help="Importance (0-1)")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List memories")
    list_parser.add_argument("-f", "--folder", help="Filter by folder")
    list_parser.add_argument("-c", "--category", help="Filter by category")
    list_parser.add_argument("-l", "--limit", type=int, default=10, help="Max items")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search memories")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("-l", "--limit", type=int, default=5, help="Max results")
    
    # Folders command
    folders_parser = subparsers.add_parser("folders", help="Manage folders")
    folders_parser.add_argument("action", choices=["list", "create", "show"])
    folders_parser.add_argument("-n", "--name", help="Folder name")
    folders_parser.add_argument("-p", "--parent", help="Parent folder")
    folders_parser.add_argument("-d", "--description", help="Description")
    folders_parser.add_argument("--path", help="Folder path to show")
    
    # Links command
    links_parser = subparsers.add_parser("links", help="View neural links")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="System statistics")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Run maintenance")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a memory")
    delete_parser.add_argument("memory_id", help="Memory ID to delete")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    commands = {
        "add": cmd_add,
        "list": cmd_list,
        "search": cmd_search,
        "folders": cmd_folders,
        "links": cmd_links,
        "stats": cmd_stats,
        "cleanup": cmd_cleanup,
        "delete": cmd_delete
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()
