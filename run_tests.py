import insert
import search

def main():
    insertion_status = insert.main(
        user_ids=[332], 
        text_ids=[22322], 
        texts=["Stake HiLo Game Review 2024. Is HiLo Legit or Rigged? outsidergaming comgamblingslotsstake originalshilo"], 
        index_name="EverGrowingVDB", 
        emb_size=384
    )
    print(f"Insertion Status: {insertion_status}")
    search_result = search.main(
        user_id=332, 
        text="Game",
        limit=5,
        index_name="EverGrowingVDB",
        emb_size=384    
    )
    print(f"search result in JSON\n {search_result}")
    search_result = search.main(
        user_id=333, 
        text="Game",
        limit=5,
        index_name="EverGrowingVDB",
        emb_size=384    
    )
    print(f"search result in JSON:\n {search_result}")

if __name__=="__main__":
    main()